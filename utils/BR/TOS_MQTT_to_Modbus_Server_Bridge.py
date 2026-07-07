#!/usr/bin/env python3
"""
TOS MQTT -> ModbusTCP SERVER bridge (PLC = Modbus Client)

Modbus TCP server:
  bind 0.0.0.0:1502, UnitID=1
Mapping (0-based offsets):
  Coil 0 = Grind
  Coil 1 = Select
  HR   0 = Setpoint (0..4095)
  HR   1 = Heartbeat (0..4095, 10 Hz)

Safety:
  If MQTT goes stale > STALE_TIMEOUT_S -> force safe (0/0/0) continuously.
"""

import time
import json
import threading
import signal
import sys

import paho.mqtt.client as mqtt

try:
    from pymodbus.server import StartTcpServer
    from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
    from pymodbus.datastore import ModbusSequentialDataBlock
except Exception as e:
    print("✗ Could not import pymodbus server components:", e)
    sys.exit(1)

# ----------------------------
# Config
# ----------------------------
MQTT_HOST = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "TOS/ee"

MODBUS_BIND_IP = "0.0.0.0"
MODBUS_PORT = 1502
UNIT_ID = 1

HEARTBEAT_HZ = 10.0
HEARTBEAT_DT = 1.0 / HEARTBEAT_HZ

STALE_TIMEOUT_S = 0.30
SAFE_ENFORCE_HZ = 20.0
SAFE_ENFORCE_DT = 1.0 / SAFE_ENFORCE_HZ

SETPOINT_MAX = 4095
HEARTBEAT_MOD = 4096  # 0..4095

# Mapping (0-based offsets)
COIL_GRIND_OFFSET = 0
COIL_SELECT_OFFSET = 1
HR_SETPOINT_OFFSET = 0
HR_HEARTBEAT_OFFSET = 1

# ----------------------------
# Shared state
# ----------------------------
lock = threading.Lock()
stop_event = threading.Event()

last_mqtt_rx = 0.0
heartbeat = 0

def ts():
    return time.strftime("%H:%M:%S")

def clamp_setpoint(x) -> int:
    try:
        v = int(x)
    except Exception:
        v = 0
    return max(0, min(SETPOINT_MAX, v))

def bool_to_bit(v) -> int:
    return 1 if bool(v) else 0

def make_datastore():
    coils = ModbusSequentialDataBlock(0, [0, 0])     # offsets 0..1
    holding = ModbusSequentialDataBlock(0, [0, 0])   # offsets 0..1
    store = ModbusSlaveContext(co=coils, hr=holding)
    return ModbusServerContext(slaves={UNIT_ID: store}, single=False)

def _set_values(context, grind_bit: int, select_bit: int, setpoint_u16: int):
    slave = context[UNIT_ID]
    slave.setValues(1, COIL_GRIND_OFFSET, [grind_bit])
    slave.setValues(1, COIL_SELECT_OFFSET, [select_bit])
    slave.setValues(3, HR_SETPOINT_OFFSET, [setpoint_u16])

def force_safe(context, reason: str):
    with lock:
        _set_values(context, 0, 0, 0)
    # niet spammen
    # print(f"[{ts()}] ⚠ SAFE ({reason})")

def heartbeat_loop(context):
    global heartbeat
    last_print = 0.0
    while not stop_event.is_set():
        with lock:
            heartbeat = (heartbeat + 1) % HEARTBEAT_MOD
            context[UNIT_ID].setValues(3, HR_HEARTBEAT_OFFSET, [heartbeat])

        now = time.time()
        if now - last_print >= 1.0:
            print(f"[{ts()}] HB={heartbeat}")
            last_print = now

        time.sleep(HEARTBEAT_DT)

def stale_watchdog_loop(context):
    while not stop_event.is_set():
        now = time.time()
        age = now - last_mqtt_rx if last_mqtt_rx > 0 else 1e9
        if age > STALE_TIMEOUT_S:
            force_safe(context, f"MQTT stale ({age:.2f}s)")
        time.sleep(SAFE_ENFORCE_DT)

def on_connect(client, userdata, flags, rc):
    global last_mqtt_rx
    if rc == 0:
        print(f"[{ts()}] ✓ MQTT connected -> subscribing {MQTT_TOPIC}")
        client.subscribe(MQTT_TOPIC)
        last_mqtt_rx = time.time()
    else:
        print(f"[{ts()}] ✗ MQTT connect failed rc={rc}")

def on_message(client, userdata, msg):
    global last_mqtt_rx
    last_mqtt_rx = time.time()

    try:
        data = json.loads(msg.payload.decode("utf-8", errors="replace"))
        grind = bool_to_bit(data.get("Grind", False))
        select = bool_to_bit(data.get("Select", False))
        setpoint = clamp_setpoint(data.get("Setpoint", 0))

        with lock:
            _set_values(userdata["context"], grind, select, setpoint)

    except Exception as e:
        print(f"[{ts()}] ⚠ MQTT parse/write error: {e}")

def shutdown(signum=None, frame=None):
    stop_event.set()
    print(f"\n[{ts()}] Stopping bridge...")

def main():
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    context = make_datastore()

    print(f"[{ts()}] Starting bridge")
    print(f"[{ts()}] Modbus: {MODBUS_BIND_IP}:{MODBUS_PORT} unit={UNIT_ID}")
    print(f"[{ts()}] MQTT : {MQTT_HOST}:{MQTT_PORT} topic={MQTT_TOPIC}")
    print(f"[{ts()}] Map  : Coil0=Grind Coil1=Select HR0=Setpoint HR1=Heartbeat")

    threading.Thread(target=heartbeat_loop, args=(context,), daemon=True).start()
    threading.Thread(target=stale_watchdog_loop, args=(context,), daemon=True).start()

    mq = mqtt.Client(client_id="TOS_MQTT_to_Modbus_Server_Bridge")
    mq.user_data_set({"context": context})
    mq.on_connect = on_connect
    mq.on_message = on_message

    mq.connect(MQTT_HOST, MQTT_PORT, 60)
    mq.loop_start()

    StartTcpServer(context=context, address=(MODBUS_BIND_IP, MODBUS_PORT))

if __name__ == "__main__":
    main()