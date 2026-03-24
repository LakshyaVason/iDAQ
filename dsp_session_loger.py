"""
DSP Session Logger
==================
Captures raw CSV data from the C2000 DSP over UART and saves it to a
timestamped folder for offline gain-equation debugging.

Folder layout created per session:
  dsp_logs/
    YYYY-MM-DD_HH-MM-SS/
      raw_uart.csv          <- every line exactly as received from DSP
      parsed_channels.csv   <- decoded 12-channel values (engineering units)
      session_meta.json     <- port, baud, duration, sample count, errors

Usage:
  python dsp_session_logger.py                        # default port + baud
  python dsp_session_logger.py --port /dev/ttyUSB0 --baud 460800
  python dsp_session_logger.py --port COM3 --baud 115200 --duration 30

Stop early at any time with Ctrl+C – files are flushed and closed cleanly.

Dependencies: pyserial (already in requirements.txt)
"""

import argparse
import csv
import json
import os
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import serial

# ─────────────────────────────────────────────
# Defaults (mirror your .env / c2000 firmware)
# ─────────────────────────────────────────────
DEFAULT_PORT     = os.getenv("C2000_UART_PORT", "/dev/ttyTHS1")
DEFAULT_BAUD     = int(os.getenv("C2000_UART_BAUD", "460800"))
LOG_ROOT         = Path("dsp_logs")

# ─────────────────────────────────────────────
# CSV column headers (must match C firmware)
# ─────────────────────────────────────────────
RAW_HEADER = ["raw_line"]

PARSED_HEADER = [
    "sample_num",
    # Voltages (mV integer from DSP → converted to V here)
    "V1_mV", "V2_mV", "V3_mV", "V4_mV",
    "V1_V",  "V2_V",  "V3_V",  "V4_V",
    # Currents (mA integer → A)
    "I1_mA", "I2_mA", "I3_mA", "I4_mA",
    "I1_A",  "I2_A",  "I3_A",  "I4_A",
    # Temperatures (centi-°C integer → °C)
    "T1_cC", "T2_cC", "T3_cC", "T4_cC",
    "T1_C",  "T2_C",  "T3_C",  "T4_C",
]

DSP_CSV_HEADER_TOKEN = "V1_mV"   # first token of the firmware header line


# ─────────────────────────────────────────────
# Parser – mirrors c2000_serial_reader.py logic
# but keeps raw integer values alongside
# ─────────────────────────────────────────────

def parse_line(line: str, sample_num: int) -> Optional[dict]:
    """
    Parse one CSV line from the C2000.

    Expected format (firmware sends trailing comma – handled here):
      V1_mV,V2_mV,V3_mV,V4_mV,I1_mA,I2_mA,I3_mA,I4_mA,T1_cC,T2_cC,T3_cC,T4_cC

    Returns a flat dict ready to write as a CSV row, or None if unparseable.
    """
    line = line.strip()
    if not line or line.startswith(DSP_CSV_HEADER_TOKEN):
        return None

    parts = [p for p in line.split(",") if p.strip()]
    if len(parts) < 12:
        return None

    try:
        ints = [int(p) for p in parts[:12]]
    except ValueError:
        return None

    v_mV = ints[0:4]
    i_mA = ints[4:8]
    t_cC = ints[8:12]

    return {
        "sample_num": sample_num,
        # raw integer values – use these to debug gain equations
        "V1_mV": v_mV[0], "V2_mV": v_mV[1], "V3_mV": v_mV[2], "V4_mV": v_mV[3],
        # converted to SI
        "V1_V":  round(v_mV[0] / 1000.0, 4),
        "V2_V":  round(v_mV[1] / 1000.0, 4),
        "V3_V":  round(v_mV[2] / 1000.0, 4),
        "V4_V":  round(v_mV[3] / 1000.0, 4),

        "I1_mA": i_mA[0], "I2_mA": i_mA[1], "I3_mA": i_mA[2], "I4_mA": i_mA[3],
        "I1_A":  round(i_mA[0] / 1000.0, 4),
        "I2_A":  round(i_mA[1] / 1000.0, 4),
        "I3_A":  round(i_mA[2] / 1000.0, 4),
        "I4_A":  round(i_mA[3] / 1000.0, 4),

        "T1_cC": t_cC[0], "T2_cC": t_cC[1], "T3_cC": t_cC[2], "T4_cC": t_cC[3],
        "T1_C":  round(t_cC[0] / 100.0, 2),
        "T2_C":  round(t_cC[1] / 100.0, 2),
        "T3_C":  round(t_cC[2] / 100.0, 2),
        "T4_C":  round(t_cC[3] / 100.0, 2),
    }


# ─────────────────────────────────────────────
# Session logger
# ─────────────────────────────────────────────

class SessionLogger:
    def __init__(self, port: str, baud: int, duration: Optional[float] = None):
        self.port     = port
        self.baud     = baud
        self.duration = duration          # seconds; None = run until Ctrl-C

        # Timestamped session folder
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = LOG_ROOT / ts
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.raw_path    = self.session_dir / "raw_uart.csv"
        self.parsed_path = self.session_dir / "parsed_channels.csv"
        self.meta_path   = self.session_dir / "session_meta.json"

        # State
        self._running      = False
        self._serial: Optional[serial.Serial] = None
        self._sample_count = 0
        self._parse_errors = 0
        self._start_time   = 0.0
        self._stop_event   = threading.Event()

    # ── file handles ──────────────────────────────────────────────────

    def _open_files(self):
        self._raw_fh    = open(self.raw_path,    "w", newline="", encoding="utf-8")
        self._parsed_fh = open(self.parsed_path, "w", newline="", encoding="utf-8")
        self._raw_writer    = csv.DictWriter(self._raw_fh,    fieldnames=RAW_HEADER)
        self._parsed_writer = csv.DictWriter(self._parsed_fh, fieldnames=PARSED_HEADER)
        self._raw_writer.writeheader()
        self._parsed_writer.writeheader()

    def _close_files(self):
        self._raw_fh.flush()
        self._parsed_fh.flush()
        self._raw_fh.close()
        self._parsed_fh.close()

    # ── metadata ──────────────────────────────────────────────────────

    def _write_meta(self, stopped_by: str):
        elapsed = time.time() - self._start_time
        meta = {
            "port":           self.port,
            "baud":           self.baud,
            "session_folder": str(self.session_dir),
            "start_time":     datetime.fromtimestamp(self._start_time).isoformat(),
            "elapsed_s":      round(elapsed, 2),
            "sample_count":   self._sample_count,
            "parse_errors":   self._parse_errors,
            "stopped_by":     stopped_by,
            "files": {
                "raw_uart":        str(self.raw_path),
                "parsed_channels": str(self.parsed_path),
            },
            "column_notes": {
                "V1–V4_mV":  "Raw integer millivolts from DSP (use to check gain equations)",
                "V1–V4_V":   "Converted to volts (mV / 1000)",
                "I1–I4_mA":  "Raw integer milliamps from DSP",
                "I1–I4_A":   "Converted to amps (mA / 1000)",
                "T1–T4_cC":  "Raw integer centi-degC from DSP (use to verify NTC equation)",
                "T1–T4_C":   "Converted to degC (cC / 100)",
            },
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))

    # ── read loop ─────────────────────────────────────────────────────

    def _read_loop(self):
        line_buf = ""
        flush_counter = 0

        while not self._stop_event.is_set():
            # Duration guard
            if self.duration and (time.time() - self._start_time) >= self.duration:
                print(f"\n⏱  Duration {self.duration}s reached – stopping.")
                self._stop_event.set()
                break

            try:
                waiting = self._serial.in_waiting
            except Exception:
                print("\n⚠  Serial port error – stopping.")
                self._stop_event.set()
                break

            if waiting:
                chunk = self._serial.read(waiting)
                line_buf += chunk.decode("utf-8", errors="ignore")

                while "\n" in line_buf:
                    line, line_buf = line_buf.split("\n", 1)
                    raw_line = line.strip()

                    # Always write raw line
                    self._raw_writer.writerow({"raw_line": raw_line})

                    # Skip header / empty
                    if not raw_line or raw_line.startswith(DSP_CSV_HEADER_TOKEN):
                        continue

                    parsed = parse_line(raw_line, self._sample_count + 1)
                    if parsed:
                        self._sample_count += 1
                        self._parsed_writer.writerow(parsed)

                        # Progress ticker
                        if self._sample_count % 500 == 0:
                            elapsed = time.time() - self._start_time
                            rate = self._sample_count / elapsed if elapsed else 0
                            print(
                                f"\r  samples={self._sample_count:>8,}  "
                                f"rate={rate:>6.0f} Hz  "
                                f"errors={self._parse_errors}   ",
                                end="", flush=True
                            )
                    else:
                        self._parse_errors += 1

                    # Flush to disk periodically so data survives a hard kill
                    flush_counter += 1
                    if flush_counter >= 200:
                        self._raw_fh.flush()
                        self._parsed_fh.flush()
                        flush_counter = 0
            else:
                time.sleep(0.001)   # yield CPU when port is quiet

    # ── public API ────────────────────────────────────────────────────

    def start(self):
        print(f"\n{'='*60}")
        print(f"  DSP Session Logger")
        print(f"{'='*60}")
        print(f"  Port      : {self.port}")
        print(f"  Baud      : {self.baud}")
        print(f"  Session   : {self.session_dir}")
        print(f"  Duration  : {'unlimited (Ctrl-C to stop)' if not self.duration else f'{self.duration}s'}")
        print(f"{'='*60}\n")

        # Open serial
        try:
            self._serial = serial.Serial(
                port=self.port, baudrate=self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
            )
        except serial.SerialException as exc:
            print(f"❌  Cannot open {self.port}: {exc}")
            sys.exit(1)

        self._open_files()
        self._start_time = time.time()
        self._running    = True

        # Graceful Ctrl-C
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("✅  Serial open – logging started. Press Ctrl-C to stop.\n")
        self._read_loop()
        self._finish("duration_elapsed" if self.duration else "stop_event")

    def _signal_handler(self, signum, frame):
        print("\n\n🛑  Interrupt received – flushing and closing…")
        self._stop_event.set()

    def _finish(self, stopped_by: str):
        self._close_files()
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._write_meta(stopped_by)
        elapsed = time.time() - self._start_time

        print(f"\n{'='*60}")
        print(f"  Session complete")
        print(f"{'='*60}")
        print(f"  Samples logged : {self._sample_count:,}")
        print(f"  Parse errors   : {self._parse_errors}")
        print(f"  Duration       : {elapsed:.1f}s")
        print(f"  Avg sample rate: {self._sample_count/elapsed:.0f} Hz" if elapsed else "")
        print(f"\n  Files saved to : {self.session_dir}/")
        print(f"    raw_uart.csv        – every raw UART line")
        print(f"    parsed_channels.csv – decoded 12-ch with raw ints + SI units")
        print(f"    session_meta.json   – port/baud/timing/error summary")
        print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Log C2000 DSP CSV data to a timestamped folder for gain-equation debugging."
    )
    parser.add_argument("--port",     default=DEFAULT_PORT, help=f"Serial port (default: {DEFAULT_PORT})")
    parser.add_argument("--baud",     default=DEFAULT_BAUD, type=int, help=f"Baud rate (default: {DEFAULT_BAUD})")
    parser.add_argument("--duration", default=None,         type=float,
                        help="Stop after N seconds (default: run until Ctrl-C)")
    args = parser.parse_args()

    logger = SessionLogger(port=args.port, baud=args.baud, duration=args.duration)
    logger.start()


if __name__ == "__main__":
    main()