#!/usr/bin/env python3
"""
test_serial_connection.py

Tests the C2000 binary packet stream from the Jetson side.
Run this before starting the main server to confirm:
  1. Port opens at 921600 baud
  2. Sync bytes 0xA5 0x5A are detected
  3. CRC validates correctly
  4. All 12 channels print non-zero values

Usage:
    python3 test_serial_connection.py
    python3 test_serial_connection.py --port /dev/ttyTHS2
    python3 test_serial_connection.py --port /dev/ttyUSB0 --baud 921600
"""

import argparse
import os
import struct
import sys
import time

import serial


# ── Packet constants ───────────────────────────────────────────────────────────
SYNC0          = 0xA5
SYNC1          = 0x5A
PKT_TOTAL_LEN  = 60
PKT_HEADER_LEN = 6
PKT_PAYLOAD    = struct.Struct("<I 4i 4i 4i")   # sampleCount + vin[4] + i[4] + t[4]

import os
from dotenv import load_dotenv

load_dotenv()  # loads .env from current working directory
print(os.getenv("C2000_UART_BAUD"))
def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def decode_packet(packet: bytes) -> dict:
    sc, v0, v1, v2, v3, i0, i1, i2, i3, t0, t1, t2, t3 = PKT_PAYLOAD.unpack_from(packet, PKT_HEADER_LEN)
    return {
        "sample_count": sc,
        "voltage":     [v0/1000, v1/1000, v2/1000, v3/1000],
        "current":     [i0/1000, i1/1000, i2/1000, i3/1000],
        "temperature": [t0/100,  t1/100,  t2/100,  t3/100 ],
    }


def test_connection(port: str, baudrate: int, duration: int = 10):
    print()
    print("=" * 60)
    print("  iDAQ C2000 Binary Packet — Connection Test")
    print("=" * 60)
    print(f"  Port     : {port}")
    print(f"  Baud     : {baudrate}")
    print(f"  Duration : {duration} s")
    print("=" * 60)

    # ── Step 1: Open port ──────────────────────────────────────────────────────
    print("\n[1/4] Opening serial port...")
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=2.0,
        )
        print(f"      ✅ Port open: {ser.name}")
    except Exception as e:
        print(f"      ❌ Failed to open port: {e}")
        print()
        print("  Troubleshooting:")
        print("    sudo usermod -aG dialout $USER   (then re-login)")
        print("    sudo systemctl stop nvgetty")
        print("    sudo systemctl disable nvgetty")
        sys.exit(1)

    # ── Step 2: Check raw bytes arriving ──────────────────────────────────────
    print("\n[2/4] Checking raw bytes (2 s)...")
    time.sleep(0.1)
    raw = ser.read(256)
    if not raw:
        print("      ❌ No bytes received")
        print()
        print("  Troubleshooting:")
        print("    - Is C2000 powered on and firmware running?")
        print("    - Check TX wire: C2000 GPIO29 → Jetson Pin 10 (RXD)")
        print("    - Check GND is shared between boards")
        ser.close()
        sys.exit(1)

    print(f"      ✅ {len(raw)} bytes received")
    print(f"      First 16 bytes: {raw[:16].hex(' ')}")

    # ── Step 3: Find sync bytes ────────────────────────────────────────────────
    print("\n[3/4] Searching for sync pattern 0xA5 0x5A...")
    raw += ser.read(512)   # grab more data to find sync
    sync_found = False
    for i in range(len(raw) - 1):
        if raw[i] == SYNC0 and raw[i+1] == SYNC1:
            sync_found = True
            print(f"      ✅ Sync found at byte offset {i}")
            break

    if not sync_found:
        print("      ❌ Sync bytes 0xA5 0x5A not found in received data")
        print()
        print("  This means the C2000 is NOT running the new binary firmware.")
        print("  The old CSV firmware is still flashed.")
        print()
        print("  Action: Flash the new binary firmware in Code Composer Studio")
        print("          then re-run this test.")
        print()
        print(f"  Raw data received ({len(raw)} bytes):")
        print(f"  {raw[:64].hex(' ')}")
        ser.close()
        sys.exit(1)

    # ── Step 4: Decode live packets ────────────────────────────────────────────
    print(f"\n[4/4] Decoding packets for {duration} s...\n")
    print(f"  {'pkts':>6}  {'crc_err':>7}  {'V[0..3] V':^36}  {'I[0..3] A':^36}  {'T[0..3] °C':^36}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*36}  {'-'*36}  {'-'*36}")

    buf       = bytearray(raw)   # seed with data already read
    pkts      = 0
    crc_errs  = 0
    deadline  = time.time() + duration

    while time.time() < deadline:
        # Top up buffer
        waiting = ser.in_waiting
        if waiting:
            buf.extend(ser.read(waiting))
        else:
            time.sleep(0.01)
            continue

        # Process complete packets
        while len(buf) >= PKT_TOTAL_LEN:
            # Find sync
            pos = -1
            for i in range(len(buf) - 1):
                if buf[i] == SYNC0 and buf[i+1] == SYNC1:
                    pos = i
                    break
            if pos == -1:
                buf = buf[-1:]
                break
            if pos > 0:
                buf = buf[pos:]
            if len(buf) < PKT_TOTAL_LEN:
                break

            pkt = bytes(buf[:PKT_TOTAL_LEN])

            # CRC check
            exp = struct.unpack_from("<H", pkt, 58)[0]
            got = crc16_ccitt_false(pkt[:58])
            if got != exp:
                crc_errs += 1
                buf = buf[2:]
                continue

            buf = buf[PKT_TOTAL_LEN:]
            d   = decode_packet(pkt)
            pkts += 1

            # Print every 500th packet (~every 0.05 s at 10 kHz) to avoid spam
            if pkts % 500 == 0 or pkts <= 5:
                v = [f"{x:7.3f}" for x in d["voltage"]]
                i = [f"{x:7.3f}" for x in d["current"]]
                t = [f"{x:6.2f}" for x in d["temperature"]]
                print(f"  {pkts:>6}  {crc_errs:>7}  {' '.join(v)}  {' '.join(i)}  {' '.join(t)}")

    ser.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Test Summary")
    print("=" * 60)
    rate = pkts / duration
    print(f"  Packets received : {pkts}")
    print(f"  CRC errors       : {crc_errs}")
    print(f"  Packet rate      : {rate:.0f} Hz  (firmware target: 10000 Hz)")
    print()

    if pkts == 0:
        print("  ❌ FAIL — no valid packets decoded")
        print("     Check firmware is flashed and C2000 is running")
    elif crc_errs > pkts * 0.01:
        print("  ⚠  WARN — CRC error rate > 1%")
        print("     Check cable quality and length (keep under 1 m)")
    else:
        print("  ✅ PASS — binary stream healthy")
        if rate < 100:
            print(f"  ⚠  Packet rate ({rate:.0f} Hz) is low — check SAMPLE_DT_US in firmware")
        else:
            print(f"  ✅ Packet rate looks good ({rate:.0f} Hz)")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test C2000 binary UART stream")
    parser.add_argument("--port", default=os.getenv("SERIAL_PORT", "/dev/ttyTHS1"))
    parser.add_argument("--baud", type=int, default=int(os.getenv("SERIAL_BAUDRATE", "921600")))
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")
    args = parser.parse_args()

    test_connection(args.port, args.baud, args.duration)