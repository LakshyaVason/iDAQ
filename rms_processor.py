"""
RMS signal processor for iDAQ.

For each 60 Hz power cycle window, computes:
  - RMS:  sqrt(mean(x^2))  — true power representation for AC signals
  - DC:   mean(x)          — DC offset / average level
  - Peak: max(|x|)         — absolute peak value

This fixes the aliasing/Nyquist problem where raw instantaneous samples
at 0.5 Hz polling looked noisy/random on the chart.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

GRID_FREQ_HZ    = 60.0
CYCLE_DURATION_S = 1.0 / GRID_FREQ_HZ   # 16.67 ms per cycle


def compute_cycle_rms(
    df: pd.DataFrame,
    time_col: str = "TIME",
    signal_cols: List[str] = None,
    cycles_per_window: int = 1
) -> pd.DataFrame:
    """
    Group raw samples into complete 60 Hz cycle windows and compute
    RMS, DC offset, and peak for each channel.

    Parameters
    ----------
    df                : raw DataFrame with a time column (in seconds)
    time_col          : name of the time column
    signal_cols       : columns to process (None = all numeric except time_col)
    cycles_per_window : number of 60 Hz cycles to average per output point
                        (1 = one output per 16.67 ms, 3 = one per 50 ms, etc.)

    Returns
    -------
    DataFrame with columns:
        time, {col}_rms, {col}_dc, {col}_peak  for each signal column
    """
    if signal_cols is None:
        signal_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != time_col
        ]

    t       = df[time_col].values
    t_min   = t.min()
    window_s = CYCLE_DURATION_S * cycles_per_window

    df = df.copy()
    df["_cycle"] = ((t - t_min) / window_s).astype(int)

    results = []
    for cycle_id, group in df.groupby("_cycle"):
        row = {"time": round(float(t_min + cycle_id * window_s), 6)}
        for col in signal_cols:
            vals = group[col].dropna().values
            if len(vals) == 0:
                row[f"{col}_rms"]  = 0.0
                row[f"{col}_dc"]   = 0.0
                row[f"{col}_peak"] = 0.0
                continue
            row[f"{col}_rms"]  = round(float(np.sqrt(np.mean(vals ** 2))), 4)
            row[f"{col}_dc"]   = round(float(np.mean(vals)),                4)
            row[f"{col}_peak"] = round(float(np.max(np.abs(vals))),         4)
        results.append(row)

    return pd.DataFrame(results)


def get_rms_summary(rms_df: pd.DataFrame, col_prefix: str) -> Dict:
    """Return latest RMS/DC/peak values for a given channel prefix."""
    if len(rms_df) == 0:
        return {"rms": 0.0, "dc": 0.0, "peak": 0.0}
    latest = rms_df.iloc[-1]
    return {
        "rms":  float(latest.get(f"{col_prefix}_rms",  0)),
        "dc":   float(latest.get(f"{col_prefix}_dc",   0)),
        "peak": float(latest.get(f"{col_prefix}_peak", 0)),
    }