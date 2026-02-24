"""
Batch dataset manager for iDAQ — RandomForest classifier only.

Scans training_data/ for F0-F7 prefixed CSV files, combines them into
a labeled dataset, and trains a single RandomForest that classifies all
fault types (F0 = Normal, F1-F7 = specific faults).

File naming convention:
    training_data/
        F0_normal_run1.csv
        F0_normal_run2.csv
        F1_overvoltage.csv
        F2_overcurrent.csv
        ...etc

Admin panel triggers: POST /auto-train
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

TRAINING_DIR    = Path("training_data")
ARTIFACTS_DIR   = Path("artifacts")
CLASSIFIER_PATH = ARTIFACTS_DIR / "fault_classifier.joblib"
FEATURE_COLS    = ["Vin", "Iin", "MOSFET_Vds", "SCR_Vds"]

# Edit these to match your actual fault definitions
FAULT_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "Fault Type 1",
    2: "Fault Type 2",
    3: "Fault Type 3",
    4: "Fault Type 4",
    5: "Fault Type 5",
    6: "Fault Type 6",
    7: "Fault Type 7",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _get_fault_label(filename: str) -> int:
    """Extract fault class 0-7 from filename prefix (e.g. 'F3_short.csv' → 3)."""
    match = re.match(r"F(\d)", Path(filename).name, re.IGNORECASE)
    return int(match.group(1)) if match else -1


def load_all_csvs(training_dir: Path = TRAINING_DIR) -> pd.DataFrame:
    """
    Load and label every F0-F7 CSV in training_dir.

    Returns a combined DataFrame with a 'fault_type' column (int 0-7)
    and a 'source_file' column for traceability.
    """
    frames = []
    csv_files = sorted(training_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{training_dir}'. "
            "Create the folder and add F0_*.csv … F7_*.csv files."
        )

    for csv_path in csv_files:
        label = _get_fault_label(csv_path.name)
        if label < 0:
            print(f"  ⚠  Skipping '{csv_path.name}' — no F0-F7 prefix")
            continue

        df = pd.read_csv(csv_path)
        df["fault_type"]  = label
        df["source_file"] = csv_path.name
        frames.append(df)
        fault_name = FAULT_NAMES.get(label, f"Class {label}")
        print(f"  ✓  {csv_path.name}: {len(df)} rows → {fault_name}")

    if not frames:
        raise ValueError(
            "No labeled CSVs loaded. Ensure files start with F0–F7."
        )

    combined = pd.concat(frames, ignore_index=True)
    dist = combined["fault_type"].value_counts().sort_index()
    print(f"\n  Total: {len(combined)} rows across {len(frames)} files")
    print(f"  Class distribution:\n{dist.to_string()}\n")
    return combined


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(
    df: pd.DataFrame,
) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train a RandomForest on all fault classes.

    F0 = Normal (class 0), F1-F7 = fault types.
    Uses class_weight='balanced' to handle unequal sample counts.
    Runs 5-fold stratified CV and reports accuracy + F1-macro.

    Returns
    -------
    (fitted_classifier, metrics_dict)
    """
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    if not feature_cols:
        raise ValueError(
            f"None of the expected feature columns {FEATURE_COLS} "
            f"found in data. Got: {list(df.columns)}"
        )

    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df["fault_type"].astype(int)

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    # 5-fold stratified cross-validation
    cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results  = cross_validate(
        clf, X, y, cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=False,
    )

    # Fit final model on the full dataset
    clf.fit(X, y)

    class_dist   = y.value_counts().sort_index().to_dict()
    present_names = {int(k): FAULT_NAMES.get(int(k), f"Class {k}")
                     for k in y.unique()}

    metrics = {
        "features":             feature_cols,
        "classes":              sorted(y.unique().tolist()),
        "fault_names":          present_names,
        "n_samples":            int(len(df)),
        "class_distribution":   {int(k): int(v) for k, v in class_dist.items()},
        "cv_accuracy_mean":     round(float(cv_results["test_accuracy"].mean()), 4),
        "cv_accuracy_std":      round(float(cv_results["test_accuracy"].std()),  4),
        "cv_f1_macro_mean":     round(float(cv_results["test_f1_macro"].mean()), 4),
        "cv_f1_macro_std":      round(float(cv_results["test_f1_macro"].std()),  4),
    }
    return clf, metrics


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_auto_training(training_dir: Path = TRAINING_DIR) -> Dict:
    """
    Full pipeline: scan training_data/ → train classifier → save artifact.

    Called by POST /auto-train in main.py.

    Returns
    -------
    dict with status, classifier metrics, and artifact path.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    print("[AutoTrain] Scanning training_data/ ...")
    df = load_all_csvs(training_dir)

    print("[AutoTrain] Training RandomForest ...")
    clf, metrics = train_classifier(df)

    # Save classifier + feature list + fault names in one artifact
    joblib.dump(
        {
            "model":       clf,
            "features":    metrics["features"],
            "fault_names": metrics["fault_names"],
        },
        CLASSIFIER_PATH,
    )

    print(
        f"[AutoTrain] ✅ Done.\n"
        f"  CV accuracy : {metrics['cv_accuracy_mean']:.3f} "
        f"(±{metrics['cv_accuracy_std']:.3f})\n"
        f"  CV F1-macro : {metrics['cv_f1_macro_mean']:.3f} "
        f"(±{metrics['cv_f1_macro_std']:.3f})\n"
        f"  Saved to    : {CLASSIFIER_PATH}"
    )

    return {
        "status":     "success",
        "classifier": metrics,
        "artifact":   str(CLASSIFIER_PATH),
    }


# ---------------------------------------------------------------------------
# Load helper (used by ai_agent.py)
# ---------------------------------------------------------------------------

def load_classifier_artifact() -> Tuple[RandomForestClassifier, List[str], Dict]:
    """
    Load the saved classifier, feature list, and fault name map.

    Returns
    -------
    (model, feature_columns, fault_names_dict)
    """
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"No classifier at '{CLASSIFIER_PATH}'. "
            "Run auto-training first via the admin panel."
        )
    artifact = joblib.load(CLASSIFIER_PATH)
    return (
        artifact["model"],
        artifact["features"],
        artifact.get("fault_names", {}),
    )


if __name__ == "__main__":
    result = run_auto_training()
    print(json.dumps(result, indent=2))