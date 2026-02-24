"""
Live data loader for streaming real CSV data with RMS signal processing.

Groups data by unique TIME values, computes per-cycle RMS/DC/peak,
then streams the processed values so the chart shows smooth, meaningful
trends instead of aliased noise.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from data_adapter import PowerElectronicsDataAdapter
from rms_processor import compute_cycle_rms


class LiveDataLoader:
    """Loads and streams real CSV data, processed into per-cycle RMS values."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.adapter  = PowerElectronicsDataAdapter(
            r_thermal_mosfet=1.0,
            r_thermal_scr=0.8,
            t_ambient=25.0
        )
        self.data: List[Dict]      = []
        self.unique_times: List[float] = []
        self.current_index: int    = 0
        self.load_data()

    def load_data(self):
        """Load CSV, compute RMS per 60 Hz cycle, prepare streaming buffer."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(df)} rows from {self.csv_path.name}")
        print(f"   Columns: {list(df.columns)}")

        # Get unique TIME values
        self.unique_times = sorted(df["TIME"].unique())
        print(f"   Unique TIME values: {len(self.unique_times)}")
        print(f"   TIME range: {self.unique_times[0]:.4f}s to {self.unique_times[-1]:.4f}s")

        # ── RMS processing ─────────────────────────────────────────────────
        # Use the raw df directly for RMS (all rows, not just first per time)
        rms_df = compute_cycle_rms(
            df,
            time_col="TIME",
            signal_cols=["Vin", "Iin", "MOSFET_Vds", "SCR_Vds"],
            cycles_per_window=1   # one output point per 16.67 ms cycle
        )
        print(f"✅ RMS computed: {len(rms_df)} cycle windows")

        # Also compute temperatures from the adapter (one row per unique TIME)
        grouped_data = []
        for time_val in self.unique_times:
            row = df[df["TIME"] == time_val].iloc[0]
            grouped_data.append({
                "TIME":        time_val,
                "Vin":         row["Vin"],
                "Iin":         row["Iin"],
                "MOSFET_Vds":  row["MOSFET_Vds"],
                "SCR_Vds":     row["SCR_Vds"],
            })

        grouped_df = pd.DataFrame(grouped_data)
        idaq_df    = self.adapter.convert_to_idaq_format(grouped_df)

        # ── Build streaming buffer from RMS data ───────────────────────────
        # rms_df rows are indexed by cycle; map temperatures from idaq_df
        # by nearest time index
        n = min(len(rms_df), len(idaq_df))
        self.data = []

        for i in range(n):
            rms_row  = rms_df.iloc[i]
            temp_row = idaq_df.iloc[min(i, len(idaq_df) - 1)]

            self.data.append({
                # V1=Vin RMS, V2=MOSFET Vds RMS, V3=SCR Vds RMS, V4=Vin DC
                "voltage": [
                    float(rms_row.get("Vin_rms",        0.0)),
                    float(rms_row.get("MOSFET_Vds_rms", 0.0)),
                    float(rms_row.get("SCR_Vds_rms",    0.0)),
                    float(rms_row.get("Vin_dc",         0.0)),
                ],
                # C1=Iin RMS, C2=Iin DC, C3=Iin Peak
                "current": [
                    float(rms_row.get("Iin_rms",  0.0)),
                    float(rms_row.get("Iin_dc",   0.0)),
                    float(rms_row.get("Iin_peak", 0.0)),
                    0.0,
                ],
                # Temperatures don't need RMS — use thermal model values
                "temperature": temp_row["temperature"],
            })

        print(f"✅ Streaming buffer ready: {len(self.data)} points")

        # Show sample values to confirm variation
        if len(self.data) >= 5:
            print("\n📊 RMS data preview:")
            indices = [0, len(self.data)//4, len(self.data)//2,
                       3*len(self.data)//4, len(self.data)-1]
            for idx in indices:
                d = self.data[idx]
                print(f"   [{idx}] Vin_rms={d['voltage'][0]:.3f}V  "
                      f"Iin_rms={d['current'][0]:.3f}A  "
                      f"Vin_dc={d['voltage'][3]:.3f}V")

    def get_next_reading(self) -> Dict:
        """Advance to next cycle window and return its processed values."""
        if not self.data:
            raise ValueError("No data loaded")
        reading = self.data[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.data)
        return {
            "voltage":     reading["voltage"],
            "current":     reading["current"],
            "temperature": reading["temperature"],
        }

    def reset(self):
        """Reset playback to the beginning."""
        self.current_index = 0

    def get_data_info(self) -> Dict:
        """Return metadata about the loaded dataset."""
        if not self.data:
            return {"loaded": False, "error": "No data loaded"}

        all_v  = [v for d in self.data for v in d["voltage"]     if v != 0]
        all_c  = [c for d in self.data for c in d["current"]     if c != 0]
        all_t  = [t for d in self.data for t in d["temperature"] if t > 0]

        def _stats(vals):
            if not vals:
                return {"min": 0, "max": 0, "avg": 0}
            return {
                "min": round(min(vals),        2),
                "max": round(max(vals),        2),
                "avg": round(np.mean(vals),    2),
            }

        return {
            "loaded":           True,
            "total_points":     len(self.data),
            "unique_timestamps": len(self.unique_times),
            "current_index":    self.current_index,
            "source_file":      self.csv_path.name,
            "time_range": {
                "start": self.unique_times[0]  if self.unique_times else 0,
                "end":   self.unique_times[-1] if self.unique_times else 0,
            },
            "statistics": {
                "voltage":     _stats(all_v),
                "current":     _stats(all_c),
                "temperature": _stats(all_t),
            },
            "processing": "RMS per 60 Hz cycle",
        }


# ---------------------------------------------------------------------------
# Global instance helpers
# ---------------------------------------------------------------------------
_data_loader: Optional[LiveDataLoader] = None


def initialize_data_loader(
    csv_path: str = "VinIinMOSFETVdsSCRVds_240_ALL.csv"
) -> bool:
    """Initialize the global data loader."""
    global _data_loader

    project_root = Path(__file__).resolve().parent
    csv_file = (
        project_root / csv_path
        if not Path(csv_path).is_absolute()
        else Path(csv_path)
    )

    if csv_file.exists():
        try:
            _data_loader = LiveDataLoader(csv_file)
            return True
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False
    else:
        print(f"⚠️  CSV file not found: {csv_file}")
        return False


def get_live_data() -> Dict:
    """Get next RMS-processed reading from the loaded CSV."""
    if _data_loader:
        return _data_loader.get_next_reading()

    # Fallback simulation (no CSV loaded)
    import random, math
    t = random.random() * 2 * math.pi
    return {
        "voltage":     [round(192 * abs(math.sin(t)), 2),
                        round(random.uniform(1, 3), 2),
                        round(random.uniform(0.3, 2), 2), 0],
        "current":     [round(10.5 + random.uniform(-0.5, 0.5), 2), 0, 0, 0],
        "temperature": [round(random.uniform(30, 50), 2),
                        round(random.uniform(28, 40), 2), 25, 0],
    }


def get_loader_info() -> Dict:
    """Return info about the current data loader state."""
    if _data_loader:
        return _data_loader.get_data_info()
    return {"loaded": False, "message": "Using simulation mode"}


if __name__ == "__main__":
    print("Testing Live Data Loader (RMS processed)...")
    print("=" * 60)

    if initialize_data_loader():
        info = get_loader_info()
        print(f"\n📊 Data Info:")
        print(f"   Unique timestamps: {info['unique_timestamps']}")
        print(f"   Processing:        {info['processing']}")

        print(f"\n🔄 Sample readings (should show smooth variation):")
        for i in range(10):
            r = get_live_data()
            print(f"   {i+1}: Vin_rms={r['voltage'][0]:.3f}V  "
                  f"Iin_rms={r['current'][0]:.3f}A  "
                  f"Vin_dc={r['voltage'][3]:.3f}V")
    else:
        print("Failed to initialize — check CSV path")