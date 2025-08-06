#!/usr/bin/env python3
"""
Simple MQTT Subscriber Test - simulates a PLC connecting to your broker
"""

import paho.mqtt.client as mqtt
import json
import time

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✓ Successfully connected to MQTT broker as external device!")
        print("✓ Subscribing to TOS/ee topic...")
        client.subscribe("TOS/ee")
    else:
        print(f"✗ Failed to connect. Return code: {rc}")

def on_message(client, userdata, msg):
    try:
        # Parse the JSON payload
        data = json.loads(msg.payload.decode())
        print(f"📡 Received: Grind={data['Grind']}, Select={data['Select']}, Setpoint={data['Setpoint']}")
    except Exception as e:
        print(f"⚠ Error parsing message: {e}")

def main():
    # This simulates what a PLC would do
    broker_ip = "192.168.10.3"  # Your machine's IP
    broker_port = 1883
    
    print(f"🔗 Connecting to MQTT broker at {broker_ip}:{broker_port}")
    print("This simulates how an external PLC would connect...")
    
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="TestPLC_001")
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(broker_ip, broker_port, 60)
        client.loop_start()
        
        print("📊 Listening for messages (Press Ctrl+C to stop)...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
    except Exception as e:
        print(f"✗ Connection error: {e}")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
