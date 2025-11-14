from fastapi import FastAPI, UploadFile, File
from typing import Dict
from pathlib import Path
import pandas as pd
import uvicorn
import os
import shutil

from local_llama_agent import LocalDiagnosticsAgent, SensorSnapshot


BASE_DIR = Path(__file__).parent
UPLOADS = BASE_DIR / "uploads"
UPLOADS.mkdir(exist_ok=True)

agent = LocalDiagnosticsAgent()
app = FastAPI(title="Jetson Fault Agent")

@app.post("/upload/normal")
async def upload_normal(file: UploadFile = File(...)):
    filepath = UPLOADS / "normal.csv"
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    stats = agent.fit_anomaly_baseline(filepath)
    return {"message": f"Baseline trained with {len(stats)} features."}

@app.post("/upload/fault")
async def upload_fault(file: UploadFile = File(...)):
    filepath = UPLOADS / "fault.csv"
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    msg = agent.train_fault_classifier(filepath)
    return {"message": msg}

@app.post("/diagnose")
async def diagnose(sensor: Dict[str, float]):
    # Detect anomaly
    try:
        anomalies = agent.detect_anomaly(sensor)
    except RuntimeError as e:
        anomalies = str(e)

    # Classify fault (if available)
    try:
        fault_pred = agent.classify_fault(sensor)
    except Exception as e:
        fault_pred = str(e)

    # Ask Ollama
    summary = "Live sensor data anomaly check."
    try:
        diag = agent.generate_diagnostics(summary, sensor)
    except Exception as e:
        diag = str(e)

    return {
        "fault_classification": fault_pred,
        "anomalies": anomalies,
        "diagnostics": diag,
    }

@app.get("/")
async def root():
    return {"status": "ok", "model": agent.ollama.model, "host": agent.ollama.base_url}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
