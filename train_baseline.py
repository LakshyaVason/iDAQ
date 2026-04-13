#!/usr/bin/env python3
"""
train_baseline.py
=================
Loads all normal-operation CSV files (VinIinMOSFETVdsSCRVds_[V]_ALL.csv),
fits the iDAQ anomaly detection baseline across the full operating range,
saves the baseline to disk, and registers a plain-English operating profile
with OpenAI so the LLM chat interface knows what "normal" looks like for
this specific circuit.

Usage:
    python train_baseline.py

Expects these files in the same directory (or set DATA_DIR below):
    VinIinMOSFETVdsSCRVds_100_ALL.csv
    VinIinMOSFETVdsSCRVds_110_ALL.csv
    ...
    VinIinMOSFETVdsSCRVds_240_ALL.csv

Outputs:
    artifacts/anomaly_baseline.joblib   — baseline stats for detect_anomaly()
    artifacts/operating_profile.json    — per-voltage stats + OpenAI summary
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

load_dotenv()

# Root directory of iDAQ project (same folder this script lives in)
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "60hz"  # CSVs are in the 60hz subdirectory
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_PATH = ARTIFACTS_DIR / "anomaly_baseline.joblib"
PROFILE_PATH  = ARTIFACTS_DIR / "operating_profile.json"

# Voltage levels to look for (steps of 10 from 100 to 240)
VOLTAGE_LEVELS = list(range(100, 250, 10))

# CSV columns produced by the DSP
RAW_COLS = ["TIME", "Vin", "Iin", "MOSFET_Vds", "SCR_Vds"]

# Features the anomaly detector will score (TIME excluded)
FEATURE_COLS = ["Vin", "Iin", "MOSFET_Vds", "SCR_Vds"]

# Z-score threshold used at runtime (document it here for the profile)
Z_THRESHOLD = 3.0

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# Step 1 — Load all CSV files
# ─────────────────────────────────────────────────────────────

def load_all_files() -> tuple[pd.DataFrame, dict]:
    """
    Load every VinIinMOSFETVdsSCRVds_[V]_ALL.csv found in DATA_DIR.
    Returns:
        combined_df  — all rows concatenated, with a 'voltage_level' column added
        per_voltage  — dict mapping voltage level → per-file DataFrame
    """
    frames      = []
    per_voltage = {}
    missing     = []

    print("\n── Loading CSV files ──────────────────────────────────────")
    for v in VOLTAGE_LEVELS:
        fname = DATA_DIR / f"VinIinMOSFETVdsSCRVds_{v}_ALL.csv"
        if not fname.exists():
            missing.append(v)
            continue

        try:
            df = pd.read_csv(fname)

            # Validate expected columns
            missing_cols = [c for c in RAW_COLS if c not in df.columns]
            if missing_cols:
                print(f"  ⚠  {fname.name}: missing columns {missing_cols} — skipped")
                continue

            df = df[RAW_COLS].copy()
            df["voltage_level"] = v
            per_voltage[v] = df
            frames.append(df)
            print(f"  ✓  {fname.name}  →  {len(df):,} rows")

        except Exception as exc:
            print(f"  ✗  {fname.name}: {exc}")

    if missing:
        print(f"\n  Files not found for voltage levels: {missing}")
        print("  (This is fine — baseline uses whatever files are present)")

    if not frames:
        print("\n❌  No valid CSV files found. Check DATA_DIR and filenames.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows across all files: {len(combined):,}")
    print(f"  Voltage levels loaded: {sorted(per_voltage.keys())}")
    return combined, per_voltage


# ─────────────────────────────────────────────────────────────
# Step 2 — Fit anomaly baseline
# ─────────────────────────────────────────────────────────────

def fit_baseline(df: pd.DataFrame) -> dict:
    """
    Compute mean and std for each feature column across all normal data.
    This is what DiagnosticsAgent.detect_anomaly() uses at runtime.
    Saves to BASELINE_PATH.
    """
    print("\n── Fitting anomaly baseline ───────────────────────────────")
    stats = {}

    for col in FEATURE_COLS:
        series = df[col].dropna()
        mean   = float(series.mean())
        std    = float(series.std(ddof=0))
        # Guard: never allow std=0 (would make every reading an anomaly)
        std    = max(std, 1e-6)
        stats[col] = {"mean": mean, "std": std}
        print(f"  {col:15s}  mean={mean:+12.4f}   std={std:10.4f}")

    joblib.dump({"baseline": stats, "z_threshold": Z_THRESHOLD}, BASELINE_PATH)
    print(f"\n  ✓  Baseline saved → {BASELINE_PATH}")
    return stats


# ─────────────────────────────────────────────────────────────
# Step 3 — Build per-voltage operating profile
# ─────────────────────────────────────────────────────────────

def build_operating_profile(per_voltage: dict, global_stats: dict) -> dict:
    """
    Compute per-voltage statistics and package everything into a profile
    dict that will be sent to OpenAI and saved to disk.
    """
    print("\n── Building operating profile ─────────────────────────────")
    voltage_profiles = {}

    for v, df in sorted(per_voltage.items()):
        vp = {}
        for col in FEATURE_COLS:
            series = df[col].dropna()
            vp[col] = {
                "mean":  round(float(series.mean()), 4),
                "std":   round(float(series.std(ddof=0)), 4),
                "min":   round(float(series.min()), 4),
                "max":   round(float(series.max()), 4),
                "p5":    round(float(series.quantile(0.05)), 4),
                "p95":   round(float(series.quantile(0.95)), 4),
            }
        voltage_profiles[str(v)] = vp
        print(f"  ✓  {v}V level profiled  ({len(df):,} rows)")

    profile = {
        "system":          "iDAQ power electronics monitor",
        "circuit":         "AC-DC converter (MOSFET + SCR)",
        "frequency_hz":    60,
        "voltage_levels":  sorted(per_voltage.keys()),
        "data_label":      "normal_operation",
        "fault_type":      None,
        "z_threshold":     Z_THRESHOLD,
        "features":        FEATURE_COLS,
        "global_baseline": {
            col: {
                "mean": round(v["mean"], 4),
                "std":  round(v["std"],  4),
            }
            for col, v in global_stats.items()
        },
        "per_voltage": voltage_profiles,
        "notes": (
            "All data represents confirmed normal operation collected at 60 Hz. "
            "No fault events are present. The global baseline is used for Z-score "
            "anomaly detection at runtime (threshold Z > 3.0). "
            "Vin and Iin are signed (AC waveform); MOSFET_Vds and SCR_Vds are "
            "device voltages. The voltage_level field indicates the nominal AC "
            "input voltage in volts at which each dataset was collected."
        )
    }

    profile_json = json.dumps(profile, indent=2)
    PROFILE_PATH.write_text(profile_json)
    print(f"\n  ✓  Operating profile saved → {PROFILE_PATH}")
    return profile


# ─────────────────────────────────────────────────────────────
# Step 4 — Register profile with OpenAI
# ─────────────────────────────────────────────────────────────

SYSTEM_REGISTRATION_PROMPT = """You are an expert power electronics diagnostics assistant 
for the iDAQ monitoring system. You are being initialized with the normal operating 
profile of a specific AC-DC power converter circuit.

This profile represents CONFIRMED NORMAL OPERATION across the full voltage operating 
range of the circuit. There are no fault events in this dataset. Use this information 
to:

1. Understand what normal Vin, Iin, MOSFET_Vds, and SCR_Vds values look like for 
   this specific circuit
2. Recognize when live readings deviate significantly from these baselines
3. Provide context-aware diagnostics that reference actual measured normal ranges
4. Never invent fault type labels — this system uses only normal operation data; 
   fault_type is null because no labeled fault data has been collected yet

When a user asks about normal operating ranges, reference the per-voltage profiles. 
When asked about anomalies, compare against the global baseline statistics."""


def register_with_openai(profile: dict) -> str:
    """
    Send the operating profile to OpenAI and get a confirmation summary
    that validates the LLM understood the circuit characteristics.
    Saves the response as part of the profile for future reference.
    """
    if not OPENAI_API_KEY:
        print("\n⚠  OPENAI_API_KEY not set — skipping OpenAI registration")
        print("   The baseline is still saved locally and will work for anomaly detection.")
        return ""

    print("\n── Registering profile with OpenAI ────────────────────────")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build a concise but complete context message
    profile_summary = f"""
Operating Profile for iDAQ Circuit:
- Circuit type: AC-DC converter (MOSFET + SCR topology)
- AC frequency: 60 Hz
- Voltage levels tested: {profile['voltage_levels']} V (nominal input)
- Total data: normal operation only, no faults, fault_type = null

Global baseline statistics (mean ± std across all voltage levels):
"""
    for col, stats in profile["global_baseline"].items():
        profile_summary += f"  {col}: mean={stats['mean']:+.4f}, std={stats['std']:.4f}\n"

    profile_summary += "\nPer-voltage normal operating windows (5th–95th percentile):\n"
    for v_str, vp in profile["per_voltage"].items():
        profile_summary += f"\n  {v_str}V input:\n"
        for col, s in vp.items():
            profile_summary += (
                f"    {col}: [{s['p5']:+.2f} to {s['p95']:+.2f}]  "
                f"(mean={s['mean']:+.4f})\n"
            )

    profile_summary += f"""
Anomaly detection: Z-score threshold = {profile['z_threshold']}
Z_j = |x_j - mean_j| / std_j

Please confirm you have internalized this operating profile by summarizing:
1. The normal Vin range at 120V and 240V input levels
2. What Iin values are normal at each level  
3. What MOSFET_Vds and SCR_Vds look like during normal switching
4. What Z-score threshold will trigger an anomaly alert
5. Why fault_type is null in this dataset
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-nano"),
            messages=[
                {"role": "system", "content": SYSTEM_REGISTRATION_PROMPT},
                {"role": "user",   "content": profile_summary}
            ],
            temperature=0.2,
            #max_tokens=800
        )

        confirmation = response.choices[0].message.content
        print("\n  OpenAI confirmation:\n")
        print("  " + "\n  ".join(confirmation.splitlines()))

        # Append to saved profile
        profile["openai_confirmation"] = confirmation
        PROFILE_PATH.write_text(json.dumps(profile, indent=2))
        print(f"\n  ✓  Confirmation appended to {PROFILE_PATH}")
        return confirmation

    except Exception as exc:
        print(f"\n  ✗  OpenAI registration failed: {exc}")
        print("     Baseline is still saved locally and will work for anomaly detection.")
        return ""


# ─────────────────────────────────────────────────────────────
# Step 5 — Patch ai_agent.py anomaly_stats at runtime
# ─────────────────────────────────────────────────────────────

def verify_baseline_loads() -> bool:
    """
    Quick sanity check: reload the saved baseline and confirm it parses correctly.
    Also prints the exact Python snippet to load it in ai_agent.py at startup.
    """
    print("\n── Verifying saved baseline ───────────────────────────────")
    try:
        artifact = joblib.load(BASELINE_PATH)
        baseline = artifact["baseline"]
        threshold = artifact["z_threshold"]
        print(f"  ✓  Loaded {len(baseline)} feature baselines, threshold={threshold}")
        for col, stats in baseline.items():
            print(f"     {col}: mean={stats['mean']:+.4f}  std={stats['std']:.4f}")

        print(f"""
── How to use this baseline in ai_agent.py ────────────────
Add this to DiagnosticsAgent.__init__() or call it at startup:

    import joblib
    from pathlib import Path

    _artifact = joblib.load(Path("artifacts/anomaly_baseline.joblib"))
    self.anomaly_stats  = _artifact["baseline"]
    # Z-threshold is already defaulted to 3.0 in detect_anomaly()

Or call the existing method at startup in main.py:
    agent.fit_anomaly_baseline(Path("normal.csv"))

Even simpler — add to main.py startup_event():
    baseline_path = BASE_DIR / "artifacts" / "anomaly_baseline.joblib"
    if baseline_path.exists():
        import joblib
        art = joblib.load(baseline_path)
        agent.anomaly_stats = art["baseline"]
        logger.info(f"✅ Anomaly baseline loaded: {{len(art['baseline'])}} features")
──────────────────────────────────────────────────────────""")
        return True

    except Exception as exc:
        print(f"  ✗  Baseline verification failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         iDAQ Anomaly Baseline Training Script           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Data directory : {DATA_DIR}")
    print(f"  Artifacts dir  : {ARTIFACTS_DIR}")
    print(f"  Voltage range  : 100V – 240V (steps of 10)")
    print(f"  OpenAI key     : {'configured' if OPENAI_API_KEY else 'NOT SET — skipping registration'}")

    # 1. Load
    combined_df, per_voltage = load_all_files()

    # 2. Fit global baseline
    global_stats = fit_baseline(combined_df)

    # 3. Build profile
    profile = build_operating_profile(per_voltage, global_stats)

    # 4. Register with OpenAI
    register_with_openai(profile)

    # 5. Verify
    ok = verify_baseline_loads()

    print("\n╔══════════════════════════════════════════════════════════╗")
    if ok:
        print("║  Training complete. Baseline ready for iDAQ.        ║")
    else:
        print("║  Training finished but baseline verification failed ║")
    print("╚══════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()