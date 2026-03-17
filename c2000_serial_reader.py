"""
C2000 DSP Serial Reader for Jetson Orin Nano.
Reads 12-channel CSV data from C2000 via UART (GPIO29 TX → Jetson Pin 10 RX).

CSV Format from C2000:
time_us,vin0_mV,vin1_mV,vin2_mV,vin3_mV,i0_mA,i1_mA,i2_mA,i3_mA,t0_cC,t1_cC,t2_cC,t3_cC,dac_code
"""

import serial
import threading
from collections import deque
from typing import Dict, Optional, Callable
from pathlib import Path

import os
from dotenv import load_dotenv

load_dotenv()  # loads .env from current working directory

class C2000SerialReader:
    """Reads real-time 12-channel ADC data from C2000 DSP via UART."""
    
    def __init__(
        self,
        port: str = "/dev/ttyTHS1",  # Jetson Orin Nano UART1
        baudrate: int = int(os.getenv("C2000_UART_BAUD")),
        callback: Optional[Callable[[Dict], None]] = None
    ):
        self.port = port
        self.baudrate = baudrate
        self.callback = callback
        self.serial_conn = None
        self.running = False
        self.buffer = deque(maxlen=1000)
        self.thread = None
        
        # Stats
        self.samples_received = 0
        self.parse_errors = 0
        self.header_received = False
        self.last_dac_code = 0
    

    def parse_csv_line(self, line: str) -> Optional[Dict]:
        try:
            line = line.strip()

            # Skip header and empty lines
            if not line or line.startswith('V1_mV'):
                self.header_received = True
                return None

            parts = line.split(',')
            # Filter empty trailing parts (firmware adds trailing comma)
            parts = [p for p in parts if p.strip()]
            if len(parts) < 12:
                return None
            vin0_mV = int(parts[0])
            vin1_mV = int(parts[1])
            vin2_mV = int(parts[2])
            vin3_mV = int(parts[3])

            i0_mA = int(parts[4])
            i1_mA = int(parts[5])
            i2_mA = int(parts[6])
            i3_mA = int(parts[7])

            t0_cC = int(parts[8])
            t1_cC = int(parts[9])
            t2_cC = int(parts[10])
            t3_cC = int(parts[11])

            return {
                'voltage': [
                    round(vin0_mV / 1000.0, 2),
                    round(vin1_mV / 1000.0, 2),
                    round(vin2_mV / 1000.0, 2),
                    round(vin3_mV / 1000.0, 2),
                ],
                'current': [
                    round(i0_mA / 1000.0, 2),
                    round(i1_mA / 1000.0, 2),
                    round(i2_mA / 1000.0, 2),
                    round(i3_mA / 1000.0, 2),
                ],
                'temperature': [
                    round(t0_cC / 100.0, 1),
                    round(t1_cC / 100.0, 1),
                    round(t2_cC / 100.0, 1),
                    round(t3_cC / 100.0, 1),
                ],
            }

        except (ValueError, IndexError) as e:
            self.parse_errors += 1
            return None

    def _read_loop(self):
        """Background thread reading serial data."""
        line_buffer = ""
        
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting:
                    chunk = self.serial_conn.read(self.serial_conn.in_waiting)
                    line_buffer += chunk.decode('utf-8', errors='ignore')
                    
                    while '\n' in line_buffer:
                        line, line_buffer = line_buffer.split('\n', 1)
                        data = self.parse_csv_line(line)
                        
                        if data:
                            self.samples_received += 1
                            self.buffer.append(data)
                            
                            if self.callback:
                                self.callback(data)
                                
            except Exception as e:
                print(f"Serial read error: {e}")
    
    def start(self) -> bool:
        """Start reading from C2000."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            
            print(f"✅ Connected to C2000 on {self.port} @ {self.baudrate} baud")
            print(f"   Expecting 12-channel CSV format:")
            print(f"   time_us,vin0_mV,...,vin3_mV,i0_mA,...,i3_mA,t0_cC,...,t3_cC,dac_code")
            return True
            
        except Exception as e:
            print(f"❌ Serial connection failed: {e}")
            return False
    
    def stop(self):
        """Stop reading."""
        self.running = False
        if self.serial_conn:
            self.serial_conn.close()
    
    def send_command(self, cmd: str):
        """Send command to C2000 (if implemented on C2000 side)."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write(cmd.encode())
    
    def get_latest(self) -> Optional[Dict]:
        """Get most recent reading."""
        return self.buffer[-1] if self.buffer else None
    
    def get_recent(self, count: int = 100) -> list:
        """Get recent readings."""
        return list(self.buffer)[-count:]
    
    def get_stats(self) -> Dict:
        """Get reader statistics."""
        return {
            "samples_received": self.samples_received,
            "parse_errors": self.parse_errors,
            "buffer_size": len(self.buffer),
            "connected": self.running and self.serial_conn is not None,
            "last_dac_code": self.last_dac_code
        }


# Global instance
_c2000_reader: Optional[C2000SerialReader] = None


def initialize_c2000_reader(
    port: str = "/dev/ttyTHS1",
    baudrate: int = int(os.getenv("C2000_UART_BAUD"))
) -> bool:
    """Initialize C2000 serial reader."""
    global _c2000_reader
    _c2000_reader = C2000SerialReader(port=port, baudrate=baudrate)
    return _c2000_reader.start()


def get_c2000_data() -> Dict:
    """Get latest reading from C2000."""
    if _c2000_reader:
        data = _c2000_reader.get_latest()
        if data:
            return {
                'voltage': data['voltage'],
                'current': data['current'],
                'temperature': data['temperature']
            }
    
    # Fallback if no data
    return {
        'voltage': [0.0, 0.0, 0.0, 0.0],
        'current': [0.0, 0.0, 0.0, 0.0],
        'temperature': [25.0, 25.0, 25.0, 25.0]
    }


def is_c2000_connected() -> bool:
    """Check if C2000 is connected."""
    return _c2000_reader is not None and _c2000_reader.running


def get_c2000_stats() -> Dict:
    """Get C2000 reader statistics."""
    if _c2000_reader:
        return _c2000_reader.get_stats()
    return {"connected": False}


if __name__ == "__main__":
    print("Testing C2000 Serial Reader (12-channel mode)...")
    print("=" * 60)
    
    if initialize_c2000_reader():
        import time
        
        print("Waiting for data...")
        for i in range(10):
            time.sleep(1)
            stats = get_c2000_stats()
            data = get_c2000_data()
            print(f"  Samples: {stats['samples_received']}, "
                  f"V: {data['voltage'][0]:.2f}V, "
                  f"I: {data['current'][0]:.2f}A, "
                  f"T: {data['temperature'][0]:.1f}°C")
    else:
        print("Failed to connect")