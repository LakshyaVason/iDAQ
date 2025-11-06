"""
LangServe-based API for Power Electronics Fault Detection & Nemotron embeddings.

Features:
1. Train a RandomForestClassifier on uploaded CSV data.
2. Evaluate the classifier on test data.
3. Provide an embedding service via NVIDIA's omni-embed-nemotron-3b model.
4. Runs on Jetson Orin Nano (CUDA or CPU), with smaller footprint than Llama 3.1 8B.
"""

import os
import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from langserve import add_routes
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.tools import tool
from transformers import AutoModel, AutoTokenizer
from pydantic import BaseModel

MODEL_ID = "nvidia/omni-embed-nemotron-3b"

print(f"Loading {MODEL_ID}…")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load with 8-bit quantization for Jetson
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True,  # 8-bit quantization
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

# Create a custom embedding function
def get_embeddings(text):
    """Get embeddings from the model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Get the mean of the last hidden state
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# RandomForest classifier model path
MODEL_PATH = "trained_fault_model.pkl"

# Define tools for training/evaluating the classifier
@tool
def train_fault_classifier(file_path: str) -> str:
    """Train a RandomForest classifier on fault detection dataset."""
    df = pd.read_csv(file_path)
    if "fault_type" not in df.columns:
        return "Dataset must include a 'fault_type' column."
    X = df.drop(columns=["fault_type"], errors="ignore")
    if "Time" in X.columns:
        X = X.drop(columns=["Time"])
    y = df["fault_type"].astype(int)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols].fillna(X[numeric_cols].mean())
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    import joblib
    joblib.dump(clf, MODEL_PATH)
    return f"✅ Trained RandomForest model with {len(df)} samples."

@tool
def evaluate_fault_classifier(file_path: str) -> str:
    """Evaluate the trained RandomForest classifier."""
    import joblib
    if not os.path.exists(MODEL_PATH):
        return "No trained model found. Please train the model first."
    df = pd.read_csv(file_path)
    clf = joblib.load(MODEL_PATH)
    X_test = df.drop(columns=["fault_type"], errors="ignore")
    if "Time" in X_test.columns:
        X_test = X_test.drop(columns=["Time"])
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns
    X_test = X_test[numeric_cols].fillna(X_test[numeric_cols].mean())
    y_test = df["fault_type"].astype(int)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return f"✅ Evaluation complete.\nAccuracy: {acc:.3f}\n\n{report}"

# FastAPI app
app = FastAPI(title="DAQ Fault Detection with Nemotron Embeddings")

# Pydantic model for embed request
class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed_text(request: EmbedRequest):
    """Get embeddings for input text."""
    embeddings = get_embeddings(request.text)
    return {"embeddings": embeddings.tolist()}

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
        "description": "LangServe-based API for embeddings and fault detection",
        "model": MODEL_ID,
        "gpu_enabled": torch.cuda.is_available(),
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)