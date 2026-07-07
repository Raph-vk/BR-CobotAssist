# multi_device_monitor.py
"""
Unified GUI for AksIM‑2 encoders **plus** the RS‑485 trigger/2‑button interface.

* **Encoders** – six UDP streams with 17‑bit single‑turn frames
  (3‑byte packets). Each encoder has its own mechanical/electrical offset so
  that *0 °* in the GUI equals real‑world zero.
* **Trigger & Buttons** – one UDP stream on port 5004    ENCODERS = [
        {"port": 5006, "label": "Encoder 6 (P5006)", "offset": 80169},  # 99800 - 19631 (54 degrees worth of counts)
        {"port": 5010, "label": "Encoder 5 (P5010)", "offset": 2900},th 5‑byte frames:
  `[0xAA, LSB, MSB, BTN, CRC]`, where CRC = sum(first 4 bytes) & 0xFF.
  The 10‑bit potentiometer value (0‑1023) is visualised, and two button
  states are shown as LEDs.

Author : (Your Name)
Date   : 2025‑07‑17
"""

from __future__ import annotations

import math
import socket
import threading
import time
import tkinter as tk
from collections import deque
from typing import Dict, List

# -----------------------------------------------------------------------------
# Global protocol constants
# -----------------------------------------------------------------------------
MAX_POSITION = 1 << 17  # 131 072 counts (17‑bit)
START_BYTE   = 0xAA     # RS‑485 trigger frame start

# -----------------------------------------------------------------------------
# 1) Shared helpers – frequency estimation over sliding window
# -----------------------------------------------------------------------------
class _FreqMeter:
    """Utility to compute message frequency over the most‑recent N samples."""

    def __init__(self, maxlen: int = 100) -> None:
        self._ts = deque(maxlen=maxlen)
        self.freq_hz = 0.0

    def tick(self) -> None:
        now = time.perf_counter()
        self._ts.append(now)
        if len(self._ts) >= 2:
            dt = self._ts[-1] - self._ts[0]
            self.freq_hz = (len(self._ts) - 1) / dt if dt else 0.0


# -----------------------------------------------------------------------------
# 2) UDP listener threads
# -----------------------------------------------------------------------------
class EncoderListener(threading.Thread):
    """Receives 3‑byte AksIM‑2 frames and exposes the *raw* position."""

    def __init__(self, local_ip: str, port: int, *, packet_size: int = 3) -> None:
        super().__init__(daemon=True)
        self.port = port
        self._packet_size = packet_size
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((local_ip, port))

        self._lock = threading.Lock()
        self.raw_pos: int = 0
        self.error  : bool = False
        self.warning: bool = False
        self._running = threading.Event()
        self._freq = _FreqMeter()

    # ------------------------------------------------------------------
    def run(self) -> None:
        self._running.set()
        leftover = b""
        while self._running.is_set():
            try:
                packet = self._sock.recv(self._packet_size)
            except socket.timeout:
                continue
            except OSError:
                break
            if not packet:
                continue

            leftover += packet
            while len(leftover) >= self._packet_size:
                frame, leftover = leftover[: self._packet_size], leftover[self._packet_size :]
                self._decode(frame)

    # ------------------------------------------------------------------
    def _decode(self, frame: bytes) -> None:
        raw_val = (frame[0] << 16) | (frame[1] << 8) | frame[2]
        w_bit   = raw_val & 0x01
        e_bit   = (raw_val >> 1) & 0x01
        pos     = raw_val >> 7

        with self._lock:
            self.raw_pos = pos
            self.error   = (e_bit == 0)
            self.warning = (w_bit == 0)
            self._freq.tick()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._running.clear()
        try:
            self._sock.close()
        except OSError:
            pass

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {
                "pos"    : self.raw_pos,
                "error"  : self.error,
                "warning": self.warning,
                "freq"   : self._freq.freq_hz,
            }


class RS485Listener(threading.Thread):
    """Receives 5‑byte trigger frames: 0xAA LSB MSB BTN CRC."""

    def __init__(self, local_ip: str, port: int = 5004) -> None:
        super().__init__(daemon=True)
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((local_ip, port))

        self._lock  = threading.Lock()
        self.pot    = 0       # 0‑1023
        self.btn1   = False
        self.btn2   = False
        self._freq  = _FreqMeter()
        self._running = threading.Event()

    # ------------------------------------------------------------------
    def run(self) -> None:
        self._running.set()
        while self._running.is_set():
            try:
                data, _ = self._sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break
            self._decode(data)

    # ------------------------------------------------------------------
    def _decode(self, buf: bytes) -> None:
        if len(buf) != 5 or buf[0] != START_BYTE:
            return
        if (sum(buf[:4]) & 0xFF) != buf[4]:
            return

        pot = (buf[2] << 8) | buf[1]
        btn = buf[3]

        with self._lock:
            self.pot  = pot & 0x3FF  # 10‑bit
            self.btn1 = bool(btn & 0x01)
            self.btn2 = bool(btn & 0x02)
            self._freq.tick()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._running.clear()
        try:
            self._sock.close()
        except OSError:
            pass

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {
                "pot" : self.pot,
                "btn1": self.btn1,
                "btn2": self.btn2,
                "freq": self._freq.freq_hz,
            }


# -----------------------------------------------------------------------------
# 3) GUI panels
# -----------------------------------------------------------------------------
class EncoderPanel:
    """Small frame showing encoder angle + status."""

    _CANVAS = 100

    def __init__(self, parent: tk.Widget, title: str) -> None:
        self._frm = tk.Frame(parent, bd=2, relief=tk.RIDGE, padx=5, pady=5)

        tk.Label(self._frm, text=title, font=("Courier New", 14, "bold")).pack(anchor="w")
        self._lbl_ang = tk.Label(self._frm, text="Angle:   0.00°", font=("Courier New", 12))
        self._lbl_ang.pack(anchor="w")
        self._lbl_err = tk.Label(self._frm, text="Error:   OK", font=("Courier New", 12))
        self._lbl_err.pack(anchor="w")
        self._lbl_wrn = tk.Label(self._frm, text="Warning: OK", font=("Courier New", 12))
        self._lbl_wrn.pack(anchor="w")
        self._lbl_frq = tk.Label(self._frm, text="Freq: 0.0 Hz", font=("Courier New", 12))
        self._lbl_frq.pack(anchor="w")

        self._cv = tk.Canvas(self._frm, width=self._CANVAS, height=self._CANVAS, bg="white")
        self._cv.pack(pady=5)
        self._needle = None

    # ------------------------------------------------------------------
    def grid(self, **kw) -> None:  # passthrough to underlying frame.grid
        self._frm.grid(padx=10, pady=10, sticky="n", **kw)

    # ------------------------------------------------------------------
    def update(self, angle_deg: float, err: bool, wrn: bool, freq: float) -> None:
        self._lbl_ang.config(text=f"Angle: {angle_deg:7.2f}°")
        self._lbl_err.config(text="Error:   ACTIVE" if err else "Error:   OK", fg="red" if err else "black")
        self._lbl_wrn.config(text="Warning: ACTIVE" if wrn else "Warning: OK", fg="orange" if wrn else "black")
        self._lbl_frq.config(text=f"Freq: {freq:4.1f} Hz")
        self._draw(angle_deg)

    # ------------------------------------------------------------------
    def _draw(self, angle_deg: float) -> None:
        if self._needle is not None:
            self._cv.delete(self._needle)
        r = self._CANVAS * 0.4
        cx = cy = self._CANVAS / 2
        ang = math.radians(angle_deg)
        x = cx + r * math.sin(ang)
        y = cy - r * math.cos(ang)
        self._needle = self._cv.create_line(cx, cy, x, y, width=2, fill="blue")


class ControlPanel:
    """Panel for potentiometer + two button LEDs."""

    def __init__(self, parent: tk.Widget, title: str = "Trigger / Buttons") -> None:
        self._frm = tk.Frame(parent, bd=2, relief=tk.RIDGE, padx=5, pady=5)

        tk.Label(self._frm, text=title, font=("Courier New", 14, "bold")).pack(anchor="w")

        # Potentiometer slider (read‑only)
        self._pot_var = tk.IntVar()
        self._scl = tk.Scale(
            self._frm,
            from_=0,
            to=1023,
            orient="horizontal",
            length=300,
            variable=self._pot_var,
            showvalue=True,
            state="disabled",
        )
        self._scl.pack(pady=5)

        # Buttons
        bar = tk.Frame(self._frm)
        bar.pack(pady=5)
        self._btn1_led = tk.Label(bar, text="BTN1", width=8, bg="grey")
        self._btn1_led.pack(side="left", padx=4)
        self._btn2_led = tk.Label(bar, text="BTN2", width=8, bg="grey")
        self._btn2_led.pack(side="left", padx=4)

        # Frequency label
        self._lbl_frq = tk.Label(self._frm, text="Freq: 0.0 Hz", font=("Courier New", 12))
        self._lbl_frq.pack(anchor="w")

    # ------------------------------------------------------------------
    def grid(self, **kw) -> None:
        self._frm.grid(padx=10, pady=10, sticky="n", **kw)

    # ------------------------------------------------------------------
    def update(self, pot: int, b1: bool, b2: bool, freq: float) -> None:
        self._pot_var.set(pot)
        self._btn1_led.config(bg="green" if b1 else "grey")
        self._btn2_led.config(bg="green" if b2 else "grey")
        self._lbl_frq.config(text=f"Freq: {freq:4.1f} Hz")


# -----------------------------------------------------------------------------
# 4) Main application glue
# -----------------------------------------------------------------------------
class DeviceGUI:
    """Top‑level window aggregating encoder + RS‑485 panels."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        local_ip: str,
        encoders: List[Dict[str, int]],
        rs485_port: int = 5004,
    ) -> None:
        self._root = root
        self._threads: List[threading.Thread] = []

        # ---------- Encoders ----------
        self._enc_cfg = encoders
        self._enc_threads: List[EncoderListener] = []
        self._enc_panels: List[EncoderPanel] = []

        main = tk.Frame(root)
        main.pack(padx=10, pady=10)

        for cfg in self._enc_cfg:
            th = EncoderListener(local_ip, cfg["port"])
            th.start()
            self._enc_threads.append(th)
            self._threads.append(th)

            pnl = EncoderPanel(main, cfg["label"])
            self._enc_panels.append(pnl)

        # Place encoder panels (2×3 grid)
        for idx, pnl in enumerate(self._enc_panels):
            pnl.grid(row=idx // 3, column=idx % 3)

        # ---------- RS‑485 Trigger/Buttons ----------
        self._rs485_thread = RS485Listener(local_ip, rs485_port)
        self._rs485_thread.start()
        self._threads.append(self._rs485_thread)

        self._ctrl_panel = ControlPanel(main)
        # Span whole width (3 columns)
        self._ctrl_panel.grid(row=2, column=0, columnspan=3)

        # Periodic refresh
        self._refresh()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.title("Encoder + RS‑485 Monitor (deg / pot)")

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        # Encoders
        for cfg, th, pnl in zip(self._enc_cfg, self._enc_threads, self._enc_panels):
            snap = th.snapshot()
            corrected = (snap["pos"] - cfg["offset"]) & (MAX_POSITION - 1)
            angle = corrected * 360.0 / MAX_POSITION
            pnl.update(angle, snap["error"], snap["warning"], snap["freq"])

        # RS‑485 control
        s = self._rs485_thread.snapshot()
        self._ctrl_panel.update(s["pot"], s["btn1"], s["btn2"], s["freq"])

        self._root.after(6, self._refresh)  # ~165 Hz

    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        for th in self._threads:
            if hasattr(th, "stop"):
                th.stop()
            th.join()
        self._root.destroy()


# -----------------------------------------------------------------------------
# 5) Entry‑point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ENCODERS = [
        {"port": 5006, "label": "Encoder 6 (P5006)", "offset": 99800+19631+32718}, # sum is thus 119431 
        {"port": 5010, "label": "Encoder 5 (P5010)", "offset": 2900},
        {"port": 5008, "label": "Encoder 4 (P5008)", "offset": (103200+8750)},
        {"port": 5009, "label": "Encoder 3 (P5009)", "offset": 63100},
        {"port": 5011, "label": "Encoder 2 (P5011)", "offset": 10200},
        {"port": 5007, "label": "Encoder 1 (P5007)", "offset": 71100},
    ]

    LOCAL_IP = "192.168.10.3"  # Adapt to your NIC

    root = tk.Tk()
    DeviceGUI(root, local_ip=LOCAL_IP, encoders=ENCODERS, rs485_port=5004)
    root.mainloop()
