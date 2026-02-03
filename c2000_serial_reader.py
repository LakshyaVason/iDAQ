"""
C2000 DSP Serial Reader for Jetson Orin Nano.
Reads CSV data from C2000 via UART (GPIO36 TX → Jetson Pin 10 RX).
"""

import serial
import threading
from collections import deque
from typing import Dict, Optional, Callable
from pathlib import Path


class C2000SerialReader:
    """Reads real-time ADC data from C2000 DSP via UART."""
    
    def __init__(
        self,
        port: str = "/dev/ttyTHS1",  # Jetson Orin Nano UART1
        baudrate: int = 115200,
        callback: Optional[Callable[[Dict], None]] = None
    ):
        self.port = port
        self.baudrate = baudrate
        self.callback = callback
        self.serial_conn = None
        self.running = False
        self.buffer = deque(maxlen=1000)
        self.thread = None
        
        # ADC conversion (12-bit, 3.3V ref) [17]
        self.adc_ref_voltage = 3.3
        self.adc_resolution = 4096
        
        # Temperature estimation
        self.r_thermal = 1.0  # °C/W
        self.t_ambient = 25.0
        
        # Stats
        self.samples_received = 0
        self.parse_errors = 0
        self.header_received = False
    
    def adc_to_voltage(self, adc_count: int) -> float:
        """Convert 12-bit ADC count to voltage."""
        return (adc_count / self.adc_resolution) * self.adc_ref_voltage
    
    def parse_csv_line(self, line: str) -> Optional[Dict]:
        """
        Parse CSV line from C2000: time_us,adc0_count,adc1_count,dac_code
        """
        try:
            line = line.strip()
            
            # Skip header
            if line.startswith('time_us') or not line:
                self.header_received = True
                return None
            
            parts = line.split(',')
            if len(parts) < 4:
                return None
            
            time_us = int(parts[0])
            adc0_count = int(parts[1])
            adc1_count = int(parts[2])
            dac_code = int(parts[3])
            
            # Convert to voltages
            v_adc0 = self.adc_to_voltage(adc0_count)
            v_adc1 = self.adc_to_voltage(adc1_count)
            v_dac = self.adc_to_voltage(dac_code)
            
            # Scale for display (adjust based on your actual circuit)
            # Example: voltage divider ratio, current sense resistor, etc.
            v_scaled = v_adc0 * 100  # If using 100:1 voltage divider
            i_scaled = v_adc1 * 10   # If using 0.1 ohm shunt (10A/V)
            
            # Temperature estimation
            power = abs(v_scaled * i_scaled)
            t_estimated = self.t_ambient + (power * self.r_thermal * 0.001)
            
            return {
                'time_us': time_us,
                'time_s': time_us / 1_000_000.0,
                'raw': {
                    'adc0': adc0_count,
                    'adc1': adc1_count,
                    'dac': dac_code
                },
                'voltage': [
                    round(v_scaled, 2),
                    round(v_adc1 * 100, 2),
                    round(v_dac * 100, 2),
                    0.0
                ],
                'current': [
                    round(i_scaled, 2),
                    0.0,
                    0.0,
                    0.0
                ],
                'temperature': [
                    round(t_estimated, 1),
                    round(t_estimated * 0.9, 1),
                    self.t_ambient,
                    0.0
                ]
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
        """Send command to C2000 (S=Start, X=Stop, R=Reset)."""
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
            "connected": self.running and self.serial_conn is not None
        }


# Global instance
_c2000_reader: Optional[C2000SerialReader] = None


def initialize_c2000_reader(
    port: str = "/dev/ttyTHS1",
    baudrate: int = 115200
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
        'voltage': [0, 0, 0, 0],
        'current': [0, 0, 0, 0],
        'temperature': [25, 25, 25, 0]
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
    print("Testing C2000 Serial Reader...")
    print("=" * 60)
    
    if initialize_c2000_reader():
        import time
        
        print("Waiting for data...")
        for i in range(10):
            time.sleep(1)
            stats = get_c2000_stats()
            data = get_c2000_data()
            print(f"  Samples: {stats['samples_received']}, "
                  f"V: {data['voltage'][0]:.1f}V, "
                  f"I: {data['current'][0]:.1f}A")
    else:
        print("Failed to connect")