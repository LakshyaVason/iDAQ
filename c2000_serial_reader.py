"""
C2000 DSP Binary Packet Reader for Jetson Orin Nano.

Parses the fixed-length binary packet stream from the F28379D firmware:

  Packet format (60 bytes, little-endian):
    [0]     sync0        : 0xA5
    [1]     sync1        : 0x5A
    [2]     version      : 0x01
    [3]     msg_type     : 0x01
    [4:6]   payload_len  : uint16  (always 52)
    [6:10]  sampleCount  : uint32
    [10:26] vin_mV[4]    : int32[4]   millivolts  → divide by 1000 for V
    [26:42] i_mA[4]      : int32[4]   milliamps   → divide by 1000 for A
    [42:58] t_cC[4]      : int32[4]   centi-°C    → divide by 100 for °C
    [58:60] crc16        : uint16  CRC-16/CCITT-FALSE over bytes [0:58]

Baud rate: 921600 (must match #define BAUD_RATE in C2000 firmware)
"""

import os
import struct
import threading
from collections import deque
from typing import Callable, Dict, Optional

import serial


# ── Packet constants (must match C2000 firmware defines) ──────────────────────
SYNC0          = 0xA5
SYNC1          = 0x5A
PKT_VERSION    = 0x01
PKT_TYPE       = 0x01
PKT_TOTAL_LEN  = 60          # bytes per complete packet
PKT_HEADER_LEN = 6           # sync(2) + version(1) + type(1) + payload_len(2)
PKT_PAYLOAD_LEN = 52         # sampleCount(4) + 12×int32(48)
# Struct layout for the payload: 1×uint32 + 12×int32
_PAYLOAD_STRUCT = struct.Struct("<I 4i 4i 4i")   # little-endian


def _crc16_ccitt_false(data: bytes) -> int:
    """
    CRC-16/CCITT-FALSE
    poly=0x1021, init=0xFFFF, xorout=0x0000, refin=False, refout=False
    Matches the firmware implementation exactly.
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class C2000SerialReader:
    """
    Reads and decodes binary ADC packets from the C2000 DSP.

    Decoded fields per packet:
        voltage[0..3]     : V   (from vin_mV,  ÷ 1000)
        current[0..3]     : A   (from i_mA,    ÷ 1000)
        temperature[0..3] : °C  (from t_cC,    ÷ 100)
    """

    def __init__(
        self,
        port:     str = os.getenv("SERIAL_PORT",     "/dev/ttyTHS1"),
        baudrate: int = int(os.getenv("SERIAL_BAUDRATE", "921600")),
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        self.port     = port
        self.baudrate = baudrate
        self.callback = callback

        self._serial:  Optional[serial.Serial] = None
        self._thread:  Optional[threading.Thread] = None
        self._running  = False

        self.buffer = deque(maxlen=1000)

        # Stats
        self.packets_received = 0
        self.crc_errors       = 0
        self.sync_losses      = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Open the serial port and start the background reader thread."""
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0,
            )
            self._running = True
            self._thread  = threading.Thread(
                target=self._read_loop, daemon=True, name="C2000Reader"
            )
            self._thread.start()
            print(f"✅ C2000 reader started on {self.port} @ {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"❌ Serial connection failed: {e}")
            return False

    def stop(self):
        """Stop the reader thread and close the port."""
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()

    def get_latest(self) -> Optional[Dict]:
        """Return the most recent decoded packet, or None if buffer is empty."""
        return self.buffer[-1] if self.buffer else None

    def get_recent(self, count: int = 100) -> list:
        """Return the last `count` decoded packets."""
        return list(self.buffer)[-count:]

    def get_stats(self) -> Dict:
        return {
            "packets_received": self.packets_received,
            "crc_errors":       self.crc_errors,
            "sync_losses":      self.sync_losses,
            "buffer_size":      len(self.buffer),
            "connected":        self._running and self._serial is not None,
            "baudrate":         self.baudrate,
            "port":             self.port,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _read_loop(self):
        """
        Background thread.

        Strategy:
          1. Search the byte stream for the two-byte sync pattern 0xA5 0x5A.
          2. Once found, read the remaining 58 bytes to complete the 60-byte packet.
          3. Validate CRC over bytes [0:58].
          4. Unpack and convert payload.
          5. If sync is lost (bad CRC), increment sync_losses and re-scan.
        """
        raw = bytearray()

        while self._running:
            try:
                # Fill buffer with whatever is waiting
                if self._serial and self._serial.in_waiting:
                    raw.extend(self._serial.read(self._serial.in_waiting))
                else:
                    # Block briefly so we don't busy-spin
                    chunk = self._serial.read(1)
                    if chunk:
                        raw.extend(chunk)
                    continue

                # Process all complete packets we can find
                while len(raw) >= PKT_TOTAL_LEN:
                    # Find sync pattern
                    sync_pos = -1
                    for i in range(len(raw) - 1):
                        if raw[i] == SYNC0 and raw[i + 1] == SYNC1:
                            sync_pos = i
                            break

                    if sync_pos == -1:
                        # No sync found — discard all but last byte (could be start of sync)
                        raw = raw[-1:]
                        break

                    if sync_pos > 0:
                        # Discard bytes before sync
                        self.sync_losses += sync_pos
                        raw = raw[sync_pos:]

                    # Need a full packet
                    if len(raw) < PKT_TOTAL_LEN:
                        break

                    packet = bytes(raw[:PKT_TOTAL_LEN])

                    # Validate header fields
                    if packet[2] != PKT_VERSION or packet[3] != PKT_TYPE:
                        # Not a valid header — skip this sync and keep looking
                        raw = raw[2:]
                        continue

                    # Validate CRC (over first 58 bytes)
                    expected_crc = struct.unpack_from("<H", packet, 58)[0]
                    computed_crc = _crc16_ccitt_false(packet[:58])

                    if computed_crc != expected_crc:
                        self.crc_errors += 1
                        # Skip past this sync byte and re-scan
                        raw = raw[2:]
                        continue

                    # ── Good packet ────────────────────────────────────────
                    # Consume it from the buffer
                    raw = raw[PKT_TOTAL_LEN:]

                    decoded = self._decode_packet(packet)
                    self.packets_received += 1
                    self.buffer.append(decoded)

                    if self.callback:
                        self.callback(decoded)

            except Exception as e:
                if self._running:
                    print(f"[C2000Reader] read error: {e}")

    def _decode_packet(self, packet: bytes) -> Dict:
        """
        Unpack a validated 60-byte packet and convert to engineering units.

        Returns:
            sample_count : int
            voltage      : list[float]  V
            current      : list[float]  A
            temperature  : list[float]  °C
        """
        # Unpack payload starting at byte 6
        # Format: uint32 sampleCount, then 4×int32 vin_mV, 4×int32 i_mA, 4×int32 t_cC
        (
            sample_count,
            vin0, vin1, vin2, vin3,
            i0,   i1,   i2,   i3,
            t0,   t1,   t2,   t3,
        ) = _PAYLOAD_STRUCT.unpack_from(packet, PKT_HEADER_LEN)

        return {
            "sample_count": sample_count,
            # millivolts → volts
            "voltage": [
                round(vin0 / 1000.0, 4),
                round(vin1 / 1000.0, 4),
                round(vin2 / 1000.0, 4),
                round(vin3 / 1000.0, 4),
            ],
            # milliamps → amps
            "current": [
                round(i0 / 1000.0, 4),
                round(i1 / 1000.0, 4),
                round(i2 / 1000.0, 4),
                round(i3 / 1000.0, 4),
            ],
            # centi-°C → °C
            "temperature": [
                round(t0 / 100.0, 2),
                round(t1 / 100.0, 2),
                round(t2 / 100.0, 2),
                round(t3 / 100.0, 2),
            ],
        }


# ── Global instance helpers (used by main.py / live_data_loader.py) ───────────

_c2000_reader: Optional[C2000SerialReader] = None


def initialize_c2000_reader(
    port:     str = os.getenv("SERIAL_PORT",     "/dev/ttyTHS1"),
    baudrate: int = int(os.getenv("SERIAL_BAUDRATE", "921600")),
) -> bool:
    global _c2000_reader
    _c2000_reader = C2000SerialReader(port=port, baudrate=baudrate)
    return _c2000_reader.start()


def get_c2000_data() -> Dict:
    """Return the latest decoded packet in iDAQ format, or zeros if no data yet."""
    if _c2000_reader:
        data = _c2000_reader.get_latest()
        if data:
            return {
                "voltage":     data["voltage"],
                "current":     data["current"],
                "temperature": data["temperature"],
            }
    return {
        "voltage":     [0.0, 0.0, 0.0, 0.0],
        "current":     [0.0, 0.0, 0.0, 0.0],
        "temperature": [25.0, 25.0, 25.0, 25.0],
    }


def is_c2000_connected() -> bool:
    return _c2000_reader is not None and _c2000_reader._running


def get_c2000_stats() -> Dict:
    if _c2000_reader:
        return _c2000_reader.get_stats()
    return {"connected": False}


# ── Quick smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("C2000 Binary Packet Reader — smoke test")
    print("=" * 60)

    # Self-test: build a known packet and verify the CRC round-trip
    def _build_test_packet() -> bytes:
        buf = bytearray(60)
        buf[0] = SYNC0
        buf[1] = SYNC1
        buf[2] = PKT_VERSION
        buf[3] = PKT_TYPE
        struct.pack_into("<H", buf, 4, PKT_PAYLOAD_LEN)
        # sampleCount=1, vin_mV=[12000,0,0,0], i_mA=[500,0,0,0], t_cC=[2500,0,0,0]
        _PAYLOAD_STRUCT.pack_into(buf, PKT_HEADER_LEN,
                                   1,
                                   12000, 0, 0, 0,
                                   500,   0, 0, 0,
                                   2500,  0, 0, 0)
        crc = _crc16_ccitt_false(bytes(buf[:58]))
        struct.pack_into("<H", buf, 58, crc)
        return bytes(buf)

    test_pkt = _build_test_packet()
    reader   = C2000SerialReader.__new__(C2000SerialReader)
    decoded  = reader._decode_packet(test_pkt)
    assert decoded["voltage"][0]     == 12.0,  "Voltage conversion failed"
    assert decoded["current"][0]     == 0.5,   "Current conversion failed"
    assert decoded["temperature"][0] == 25.0,  "Temperature conversion failed"
    print("✅ Self-test passed")
    print(f"   voltage[0]     = {decoded['voltage'][0]} V   (expected 12.0)")
    print(f"   current[0]     = {decoded['current'][0]} A   (expected 0.5)")
    print(f"   temperature[0] = {decoded['temperature'][0]} °C (expected 25.0)")

    print()
    if initialize_c2000_reader():
        print("Listening for packets (10 s)…")
        for _ in range(10):
            time.sleep(1)
            stats = get_c2000_stats()
            data  = get_c2000_data()
            print(
                f"  pkts={stats['packets_received']:6d}  "
                f"crc_err={stats['crc_errors']}  "
                f"V0={data['voltage'][0]:.3f} V  "
                f"I0={data['current'][0]:.3f} A  "
                f"T0={data['temperature'][0]:.1f} °C"
            )
    else:
        print("Could not open serial port — check SERIAL_PORT env var")