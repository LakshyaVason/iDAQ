"""
Data adapter for converting colleague's CSV format to iDAQ format.

Input CSV columns:
- TIME: timestamp
- Vin: Input voltage
- Iin: Input current  
- MOSFET_Vds: MOSFET drain-source voltage
- SCR_Vds: SCR voltage

Output format:
- voltage: [Vin, MOSFET_Vds, SCR_Vds, 0]
- current: [Iin, 0, 0, 0]
- temperature: [T_mosfet, T_scr, T_ambient, 0]
"""

import pandas as pd
import numpy as np
from pathlib import Path


class PowerElectronicsDataAdapter:
    """Converts power electronics data to iDAQ format."""
    
    def __init__(
        self,
        r_thermal_mosfet: float = 1.0,  # °C/W
        r_thermal_scr: float = 0.8,      # °C/W
        t_ambient: float = 25.0          # °C
    ):
        self.r_thermal_mosfet = r_thermal_mosfet
        self.r_thermal_scr = r_thermal_scr
        self.t_ambient = t_ambient
    
    def calculate_power_and_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate power dissipation and junction temperatures.
        
        Power dissipation formulas:
        - P_mosfet = MOSFET_Vds * |Iin|
        - P_scr = SCR_Vds * |Iin|
        
        Temperature estimation (simplified thermal model):
        - T_junction = T_ambient + (P_avg * R_thermal)
        """
        # Calculate instantaneous power (W)
        df['P_mosfet'] = df['MOSFET_Vds'] * abs(df['Iin'])
        df['P_scr'] = df['SCR_Vds'] * abs(df['Iin'])
        
        # Calculate rolling average power (over 10 samples for stability)
        window_size = min(10, len(df))
        df['P_mosfet_avg'] = df['P_mosfet'].rolling(window=window_size, min_periods=1).mean()
        df['P_scr_avg'] = df['P_scr'].rolling(window=window_size, min_periods=1).mean()
        
        # Calculate junction temperatures
        df['T_mosfet'] = self.t_ambient + (df['P_mosfet_avg'] * self.r_thermal_mosfet)
        df['T_scr'] = self.t_ambient + (df['P_scr_avg'] * self.r_thermal_scr)
        
        return df
    
    def convert_to_idaq_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert to iDAQ expected format with arrays.
        
        Returns DataFrame with columns:
        - Time: timestamp
        - voltage: list of 4 voltage values
        - current: list of 4 current values  
        - temperature: list of 4 temperature values
        """
        # Validate required columns
        required_cols = ['TIME', 'Vin', 'Iin', 'MOSFET_Vds', 'SCR_Vds']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Found: {list(df.columns)}")
        
        # Calculate power and temperatures
        df = self.calculate_power_and_temperature(df)
        
        # Create iDAQ format
        idaq_data = []
        
        for _, row in df.iterrows():
            idaq_data.append({
                'Time': row['TIME'],
                # Voltage array: [Vin, MOSFET_Vds, SCR_Vds, placeholder]
                'voltage': [
                    abs(row['Vin']),
                    row['MOSFET_Vds'],
                    row['SCR_Vds'],
                    0.0
                ],
                # Current array: [Iin, placeholder, placeholder, placeholder]
                'current': [
                    abs(row['Iin']),
                    0.0,
                    0.0,
                    0.0
                ],
                # Temperature array: [T_mosfet, T_scr, T_ambient, placeholder]
                'temperature': [
                    row['T_mosfet'],
                    row['T_scr'],
                    self.t_ambient,
                    0.0
                ]
            })
        
        return pd.DataFrame(idaq_data)
    
    def process_file(self, input_path: Path, output_path: Path = None) -> pd.DataFrame:
        """
        Process a CSV file and optionally save converted format.
        
        Args:
            input_path: Path to input CSV
            output_path: Optional path to save converted CSV
            
        Returns:
            DataFrame in iDAQ format
        """
        # Read input CSV (auto-detect delimiter)
        df = pd.read_csv(input_path)
        
        # Convert to iDAQ format
        idaq_df = self.convert_to_idaq_format(df)
        
        # Save if output path provided
        if output_path:
            # Flatten arrays for CSV export
            flat_df = pd.DataFrame({
                'Time': idaq_df['Time'],
                'V1': idaq_df['voltage'].apply(lambda x: x[0]),
                'V2': idaq_df['voltage'].apply(lambda x: x[1]),
                'V3': idaq_df['voltage'].apply(lambda x: x[2]),
                'V4': idaq_df['voltage'].apply(lambda x: x[3]),
                'C1': idaq_df['current'].apply(lambda x: x[0]),
                'C2': idaq_df['current'].apply(lambda x: x[1]),
                'C3': idaq_df['current'].apply(lambda x: x[2]),
                'C4': idaq_df['current'].apply(lambda x: x[3]),
                'T1': idaq_df['temperature'].apply(lambda x: x[0]),
                'T2': idaq_df['temperature'].apply(lambda x: x[1]),
                'T3': idaq_df['temperature'].apply(lambda x: x[2]),
                'T4': idaq_df['temperature'].apply(lambda x: x[3]),
            })
            flat_df.to_csv(output_path, index=False)
            print(f"✅ Converted data saved to {output_path}")
        
        return idaq_df


def main():
    """Example usage."""
    # Initialize adapter with your thermal parameters
    adapter = PowerElectronicsDataAdapter(
        r_thermal_mosfet=1.0,  # Update with actual datasheet value
        r_thermal_scr=0.8,      # Update with actual datasheet value
        t_ambient=25.0
    )
    
    # Process your colleague's file
    input_file = Path("VinIinMOSFETVdsSCRVds_240_ALL.csv")
    output_file = Path("VinIinMOSFETVdsSCRVds_240_ALL_idaq.csv")
    
    if input_file.exists():
        print(f"Processing {input_file}...")
        idaq_df = adapter.process_file(input_file, output_file)
        
        print(f"\n✅ Conversion complete!")
        input_rows = len(pd.read_csv(input_file))
        print(f"   Input rows: {input_rows}")
        print(f"   Output rows: {len(idaq_df)}")
        print(f"\nSample converted data:")
        print(idaq_df.head())
        
        # Calculate statistics
        all_temps = []
        for temps in idaq_df['temperature']:
            all_temps.extend([t for t in temps if t > 0])
        
        print(f"\nTemperature statistics:")
        print(f"   Min: {min(all_temps):.1f}°C")
        print(f"   Max: {max(all_temps):.1f}°C")
        print(f"   Avg: {np.mean(all_temps):.1f}°C")
    else:
        print(f"❌ File not found: {input_file}")
        print("Place your CSV file in the project root directory")


