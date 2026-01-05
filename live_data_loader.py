"""
Live data loader for streaming real CSV data with waveform variation.

Groups data by unique TIME values so the graph shows actual voltage/current
changes as time progresses (e.g., -0.048s → -0.047s → ... → 0.035s).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from data_adapter import PowerElectronicsDataAdapter


class LiveDataLoader:
    """Loads and streams real CSV data grouped by unique TIME values."""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.adapter = PowerElectronicsDataAdapter(
            r_thermal_mosfet=1.0,
            r_thermal_scr=0.8,
            t_ambient=25.0
        )
        self.data: List[Dict] = []
        self.unique_times: List[float] = []
        self.current_index = 0
        self.load_data()
    
    def load_data(self):
        """Load CSV and group by unique TIME values for waveform display."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # Read CSV
        df = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(df)} rows from {self.csv_path.name}")
        print(f"   Columns found: {list(df.columns)}")
        
        # Get unique TIME values and sort them
        self.unique_times = sorted(df['TIME'].unique())
        print(f"   Unique TIME values: {len(self.unique_times)}")
        print(f"   TIME range: {self.unique_times[0]:.4f}s to {self.unique_times[-1]:.4f}s")
        
        # Group by TIME - take first row at each unique timestamp
        grouped_data = []
        for time_val in self.unique_times:
            time_slice = df[df['TIME'] == time_val]
            row = time_slice.iloc[0]  # Take first sample at this timestamp
            
            grouped_data.append({
                'TIME': time_val,
                'Vin': row['Vin'],
                'Iin': row['Iin'],
                'MOSFET_Vds': row['MOSFET_Vds'],
                'SCR_Vds': row['SCR_Vds']
            })
        
        # Convert to DataFrame and process through adapter
        grouped_df = pd.DataFrame(grouped_data)
        idaq_df = self.adapter.convert_to_idaq_format(grouped_df)
        
        # Convert to list of dicts
        self.data = idaq_df.to_dict('records')
        
        print(f"✅ Grouped into {len(self.data)} unique timesteps for streaming")
        
        # Show variation preview
        if len(self.data) >= 5:
            print(f"\n📊 Data variation preview:")
            indices = [0, len(self.data)//4, len(self.data)//2, 3*len(self.data)//4, len(self.data)-1]
            for idx in indices:
                d = self.data[idx]
                t = self.unique_times[idx]
                print(f"   t={t:.4f}s: Vin={d['voltage'][0]:.1f}V, MOSFET={d['voltage'][1]:.1f}V")
    
    def get_next_reading(self) -> Dict:
        """Get next data point (advances to next unique TIME)."""
        if not self.data:
            raise ValueError("No data loaded")
        
        reading = self.data[self.current_index]
        
        # Move to next unique timestamp, loop if at end
        self.current_index = (self.current_index + 1) % len(self.data)
        
        return {
            'voltage': reading['voltage'],
            'current': reading['current'],
            'temperature': reading['temperature']
        }
    
    def reset(self):
        """Reset to beginning."""
        self.current_index = 0
    
    def get_data_info(self) -> Dict:
        """Get information about loaded data."""
        if not self.data:
            return {"loaded": False, "error": "No data loaded"}
        
        all_voltages = []
        all_currents = []
        all_temps = []
        
        for point in self.data:
            all_voltages.extend([v for v in point['voltage'] if v != 0])
            all_currents.extend([c for c in point['current'] if c != 0])
            all_temps.extend([t for t in point['temperature'] if t > 0])
        
        return {
            "loaded": True,
            "total_points": len(self.data),
            "unique_timestamps": len(self.unique_times),
            "current_index": self.current_index,
            "source_file": str(self.csv_path.name),
            "time_range": {
                "start": self.unique_times[0] if self.unique_times else 0,
                "end": self.unique_times[-1] if self.unique_times else 0
            },
            "statistics": {
                "voltage": {
                    "min": round(min(all_voltages), 2) if all_voltages else 0,
                    "max": round(max(all_voltages), 2) if all_voltages else 0,
                    "avg": round(np.mean(all_voltages), 2) if all_voltages else 0
                },
                "current": {
                    "min": round(min(all_currents), 2) if all_currents else 0,
                    "max": round(max(all_currents), 2) if all_currents else 0,
                    "avg": round(np.mean(all_currents), 2) if all_currents else 0
                },
                "temperature": {
                    "min": round(min(all_temps), 2) if all_temps else 0,
                    "max": round(max(all_temps), 2) if all_temps else 0,
                    "avg": round(np.mean(all_temps), 2) if all_temps else 0
                }
            }
        }


# Global instance
_data_loader: Optional[LiveDataLoader] = None


def initialize_data_loader(csv_path: str = "VinIinMOSFETVdsSCRVds_240_ALL.csv") -> bool:
    """Initialize the global data loader."""
    global _data_loader
    
    project_root = Path(__file__).resolve().parent
    csv_file = project_root / csv_path if not Path(csv_path).is_absolute() else Path(csv_path)
    
    if csv_file.exists():
        try:
            _data_loader = LiveDataLoader(csv_file)
            return True
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False
    else:
        print(f"⚠️ CSV file not found: {csv_file}")
        print("   Place your CSV file in the project root")
        return False


def get_live_data() -> Dict:
    """Get next reading from loaded CSV data."""
    if _data_loader:
        return _data_loader.get_next_reading()
    else:
        # Fallback simulation
        import random
        import math
        t = random.random() * 2 * math.pi
        return {
            'voltage': [
                round(192 * abs(math.sin(t)), 2),
                round(random.choice([2, 400]), 2),
                round(random.uniform(0.3, 2), 2),
                0
            ],
            'current': [round(10.5 + random.uniform(-0.5, 0.5), 2), 0, 0, 0],
            'temperature': [round(random.uniform(30, 50), 2), round(random.uniform(28, 40), 2), 25, 0]
        }


def get_loader_info() -> Dict:
    """Get information about the data loader."""
    if _data_loader:
        return _data_loader.get_data_info()
    return {"loaded": False, "message": "Using simulation mode"}


if __name__ == "__main__":
    print("Testing Live Data Loader (Grouped by TIME)...")
    print("=" * 60)
    
    if initialize_data_loader():
        info = get_loader_info()
        print(f"\n📊 Data Info:")
        print(f"   Unique timestamps: {info['unique_timestamps']}")
        print(f"   Time range: {info['time_range']['start']:.4f}s to {info['time_range']['end']:.4f}s")
        
        print(f"\n🔄 Sample readings (should show variation):")
        for i in range(10):
            reading = get_live_data()
            print(f"   {i+1}: Vin={reading['voltage'][0]:.1f}V, MOSFET={reading['voltage'][1]:.1f}V, Iin={reading['current'][0]:.1f}A")
    else:
        print("❌ Failed to initialize")