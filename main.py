"""
LangChain + LangServe application for Power Electronics Fault Detection and LLM interaction.

Features
--------
1. Train a RandomForestClassifier on uploaded CSV data (fault detection model)
2. Query an LLM (Meta-Llama 3.1 8B) for explanations or diagnostic reasoning
3. Expose REST API endpoints for chat and ML training/evaluation
4. Runs on Jetson Nano (CUDA or CPU)
"""

import os
import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from langserve import add_routes
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from transformers import pipeline
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────────
# Load environment and model configuration
# ────────────────────────────────────────────────────────────────
load_dotenv()
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")

import os, torch

# Disable allocator options not supported on Jetson
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
torch._C._set_allocator_settings = lambda *a, **kw: None



# ────────────────────────────────────────────────────────────────
# Initialize the Hugging Face pipeline
# ────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_ID}...")
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    token=HF_TOKEN,
    device = 0,
    # dtype=torch.float32,
)
llm = HuggingFacePipeline(pipeline=pipe)

# LangChain memory + conversation chain
memory = ConversationBufferMemory()
chat_chain = ConversationChain(llm=llm, memory=memory)

# ────────────────────────────────────────────────────────────────
# Fault Detection Tool (Train / Evaluate)
# ────────────────────────────────────────────────────────────────
MODEL_PATH = "trained_fault_model.pkl"


@tool
def train_fault_classifier(file_path: str) -> str:
    """Train a RandomForest classifier on fault detection dataset."""
    print(f"Training on: {file_path}")
    df = pd.read_csv(file_path)

    if "fault_type" not in df.columns:
        return "Dataset must include a 'fault_type' column."

    X = df.drop(columns=["fault_type"], errors="ignore")
    if "Time" in X.columns:
        X = X.drop(columns=["Time"])
    y = df["fault_type"].astype(int)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols].fillna(X.mean())

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    # Save trained model
    import joblib
    joblib.dump(clf, MODEL_PATH)

    return f"✅ Trained RandomForest model with {len(df)} samples."


@tool
def evaluate_fault_classifier(file_path: str) -> str:
    """Evaluate trained RandomForest classifier on test dataset."""
    import joblib
    if not os.path.exists(MODEL_PATH):
        return "No trained model found. Please train the model first."

    df = pd.read_csv(file_path)
    clf = joblib.load(MODEL_PATH)

    X_test = df.drop(columns=["fault_type"], errors="ignore")
    if "Time" in X_test.columns:
        X_test = X_test.drop(columns=["Time"])
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns
    X_test = X_test[numeric_cols].fillna(X_test.mean())

    y_test = df["fault_type"].astype(int)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return f"✅ Evaluation complete.\nAccuracy: {acc:.3f}\n\n{report}"


# ────────────────────────────────────────────────────────────────
# FastAPI + LangServe Integration
# ────────────────────────────────────────────────────────────────
app = FastAPI(title="LLM-Powered Power Electronics Fault Detection API")

# Add LangChain chat route
add_routes(app, chat_chain, path="/chat")

# Add ML endpoints (manual file upload routes)
@app.post("/train")
async def train_file(file: UploadFile = File(...)):
    file_path = f"./uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    result = train_fault_classifier.run(file_path)
    return {"message": result}


@app.post("/evaluate")
async def evaluate_file(file: UploadFile = File(...)):
    file_path = f"./uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    result = evaluate_fault_classifier.run(file_path)
    return {"message": result}


@app.get("/")
async def root():
    return {
        "status": "running",
        "description": "LangServe-based LLM + Fault Detection API",
        "model": MODEL_ID,
        "gpu_enabled": torch.cuda.is_available(),
    }


# ────────────────────────────────────────────────────────────────
# Run: uvicorn main:app --reload
# Then open http://127.0.0.1:8000/docs
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
