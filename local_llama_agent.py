"""Local diagnostics agent for Jetson-based iDAQ deployments.

This module focuses on the *local* side of the hybrid architecture described by
the user.  It assumes the Jetson is running Ollama with the `llama3.2:1b`
model and exposes helpers for:

1. Making sure the model is installed (or at least reporting what to do).
2. Training / loading a lightweight RandomForest fault classifier.
3. Detecting anomalies using basic statistics derived from a reference dataset.
4. Retrieving structured troubleshooting knowledge from a local file.
5. Asking the local Llama model to combine the retrieved context with live
   sensor summaries and emit a step-by-step diagnostic report.

The intention is that this module can run fully offline on the Jetson.  A
separate script (not included here) would be responsible for invoking an
OpenAI-hosted model for cloud lookups when connectivity is available.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise


BASE_DIR = Path(__file__).parent
MODEL_NAME = "llama3.2:1b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
CLASSIFIER_PATH = BASE_DIR / "artifacts" / "local_fault_classifier.joblib"
KNOWLEDGE_BASE_PATH = BASE_DIR / "local_knowledge_base.json"


class OllamaNotRunningError(RuntimeError):
    """Raised when the Ollama server cannot be reached."""


class LocalKnowledgeBase:
    """Simple TF-IDF backed retriever over a JSON knowledge base."""

    def __init__(self, kb_path: Path = KNOWLEDGE_BASE_PATH):
        if not kb_path.exists():
            raise FileNotFoundError(
                f"Knowledge base file '{kb_path}' is missing. Please create it before running diagnostics."
            )
        self.entries = json.loads(kb_path.read_text())
        documents = [self._entry_to_document(entry) for entry in self.entries]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(documents)

    @staticmethod
    def _entry_to_document(entry: Dict[str, str]) -> str:
        return "\n".join(
            [
                entry.get("title", ""),
                entry.get("fault_type", ""),
                entry.get("symptoms", ""),
                entry.get("diagnostic_steps", ""),
            ]
        )

    def query(self, text: str, top_k: int = 3) -> List[Dict[str, str]]:
        if not text.strip():
            return self.entries[:top_k]
        query_vec = self.vectorizer.transform([text])
        scores = pairwise.cosine_similarity(query_vec, self.matrix)[0]
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [self.entries[idx] for idx in ranked_idx]


class OllamaClient:
    """Lightweight HTTP client for the local Ollama server."""

    def __init__(self, base_url: str = OLLAMA_HOST, model: str = MODEL_NAME):
        self.base_url = base_url.rstrip("/")
        self.model = model

    @staticmethod
    def _run_ollama_cli(args: Sequence[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ollama", *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def ensure_model_available(self) -> str:
        """Return a human readable status about the local model."""

        try:
            result = self._run_ollama_cli(["list"])
        except FileNotFoundError as exc:  # pragma: no cover - only triggered on systems without Ollama
            raise RuntimeError(
                "The 'ollama' CLI was not found. Install Ollama on the Jetson before running diagnostics."
            ) from exc

        if result.returncode != 0:
            return f"⚠️ Could not query Ollama: {result.stderr.strip()}"

        if self.model in result.stdout:
            return f"✅ Model '{self.model}' is already installed."

        return (
            f"ℹ️ Model '{self.model}' is not installed yet. Run 'ollama pull {self.model}' on the Jetson to download it."
        )

    def _post(self, path: str, payload: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        try:
            response = requests.post(url, json=payload, timeout=60)
        except requests.RequestException as exc:
            raise OllamaNotRunningError(
                f"Failed to reach Ollama at {self.base_url}. Ensure 'ollama serve' is running."
            ) from exc

        response.raise_for_status()
        return response.json()

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        data = self._post("/api/generate", payload)
        return data.get("response", "").strip()


@dataclass
class SensorSnapshot:
    voltage: float
    current: float
    temperature: float
    vibration: float

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "SensorSnapshot":
        voltage, current, temperature, vibration = values
        return cls(voltage, current, temperature, vibration)

    def to_dict(self) -> Dict[str, float]:
        return {
            "voltage": self.voltage,
            "current": self.current,
            "temperature": self.temperature,
            "vibration": self.vibration,
        }


class LocalDiagnosticsAgent:
    """Brings together the classifier, anomaly detector, and local LLM."""

    def __init__(
        self,
        knowledge_base: Optional[LocalKnowledgeBase] = None,
        ollama_client: Optional[OllamaClient] = None,
        classifier_path: Path = CLASSIFIER_PATH,
    ) -> None:
        self.kb = knowledge_base or LocalKnowledgeBase()
        self.ollama = ollama_client or OllamaClient()
        self.classifier_path = classifier_path
        self.classifier: Optional[RandomForestClassifier] = None
        self.feature_columns: List[str] = []

    # ------------------------------------------------------------------
    # Fault classification helpers
    # ------------------------------------------------------------------
    def train_fault_classifier(self, data_path: Path) -> str:
        df = pd.read_csv(data_path)
        if "fault_type" not in df.columns:
            raise ValueError("Dataset must contain a 'fault_type' column.")

        X = df.drop(columns=["fault_type"], errors="ignore")
        if "Time" in X.columns:
            X = X.drop(columns=["Time"])
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols].fillna(X[numeric_cols].mean())
        y = df["fault_type"].astype(int)

        clf = RandomForestClassifier(n_estimators=300, random_state=7, n_jobs=-1)
        clf.fit(X, y)

        self.classifier = clf
        self.feature_columns = numeric_cols

        self.classifier_path.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump({"model": clf, "features": numeric_cols}, self.classifier_path)

        return f"Trained classifier on {len(df)} samples with {len(numeric_cols)} features."

    def load_classifier(self) -> None:
        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"Classifier file '{self.classifier_path}' not found. Run 'train_fault_classifier' first."
            )
        artifact = joblib.load(self.classifier_path)
        self.classifier = artifact["model"]
        self.feature_columns = artifact["features"]

    def classify_fault(self, sensor_row: Dict[str, float]) -> str:
        if self.classifier is None:
            self.load_classifier()

        row = pd.DataFrame([sensor_row])
        row = row[self.feature_columns].fillna(0)
        prediction = int(self.classifier.predict(row)[0])
        proba = self.classifier.predict_proba(row)[0]
        confidence = float(np.max(proba))
        return f"Predicted fault_type={prediction} with confidence {confidence:.2f}"

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------
    def fit_anomaly_baseline(self, normal_data_path: Path) -> Dict[str, Dict[str, float]]:
        df = pd.read_csv(normal_data_path)
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {"mean": float(df[col].mean()), "std": float(df[col].std(ddof=0) or 1.0)}
        self.anomaly_stats = stats
        return stats

    def detect_anomaly(self, sensor_row: Dict[str, float], z_threshold: float = 3.0) -> Dict[str, float]:
        if not hasattr(self, "anomaly_stats"):
            raise RuntimeError("Call 'fit_anomaly_baseline' before detecting anomalies.")

        anomalies = {}
        for feature, value in sensor_row.items():
            if feature not in self.anomaly_stats:
                continue
            mean = self.anomaly_stats[feature]["mean"]
            std = self.anomaly_stats[feature]["std"]
            z_score = abs((value - mean) / std)
            if z_score >= z_threshold:
                anomalies[feature] = float(z_score)
        return anomalies

    # ------------------------------------------------------------------
    # Retrieval-augmented structured diagnostics
    # ------------------------------------------------------------------
    def generate_diagnostics(self, summary: str, sensor_row: Dict[str, float]) -> str:
        retrievals = self.kb.query(summary, top_k=3)
        context = json.dumps(retrievals, indent=2)
        prompt = f"""
You are an embedded diagnostics expert operating offline on a Jetson Orin Nano.
Use ONLY the context below and the provided sensor snapshot to craft a
step-by-step diagnostic playbook. Output markdown with three sections:
1. Fault Hypotheses
2. Recommended Checks
3. Next Isolation Steps

Context:
{context}

Sensor snapshot:
{json.dumps(sensor_row, indent=2)}

Live summary: {summary}
"""
        return self.ollama.generate(prompt)


def demo_run() -> None:
    """Demonstrate how to wire up the local agent end-to-end."""

    agent = LocalDiagnosticsAgent()

    print(agent.ollama.ensure_model_available())

    fault_data = BASE_DIR / "fault.csv"
    normal_data = BASE_DIR / "normal.csv"

    if fault_data.exists():
        print(agent.train_fault_classifier(fault_data))
    else:
        print("⚠️ 'fault.csv' missing – skipping classifier training.")

    if normal_data.exists():
        agent.fit_anomaly_baseline(normal_data)
    else:
        print("⚠️ 'normal.csv' missing – anomaly detection will be unavailable.")

    sample_snapshot = SensorSnapshot(300.0, 15.5, 72.0, 0.03)
    sensor_dict = sample_snapshot.to_dict()

    anomalies = {}
    if hasattr(agent, "anomaly_stats"):
        anomalies = agent.detect_anomaly(sensor_dict)
        if anomalies:
            print("⚠️ Detected anomalies:", anomalies)

    summary = "Inverter output voltage sag with concurrent rise in device temperature."

    try:
        diagnostics = agent.generate_diagnostics(summary, sensor_dict)
        print("\n=== Structured Diagnostics ===\n")
        print(diagnostics)
    except OllamaNotRunningError as exc:
        print(str(exc))


if __name__ == "__main__":
    demo_run()
