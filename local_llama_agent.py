"""
Local diagnostics agent for Jetson-based iDAQ deployments.

This module encapsulates the functionality needed for running a lightweight
diagnostic workflow on a Jetson Orin Nano.  It can train a RandomForest
classifier on labeled fault data, fit a simple anomaly detection baseline
on normal operation data, retrieve structured troubleshooting information
from a local knowledge base, and interface with a locally running Ollama
model for language-based diagnostics.  All operations are performed
entirely offline assuming the Ollama service is available locally.

Key components:

* ``LocalKnowledgeBase``: A TF‑IDF backed retriever over a JSON file of
  troubleshooting entries.  It supports free‑text queries and returns the
  top matching entries.
* ``OllamaClient``: A minimal HTTP wrapper for the local Ollama server.
  It exposes convenience methods to check if a model is installed and to
  generate text from the model via the ``/api/generate`` endpoint.
* ``LocalDiagnosticsAgent``: Orchestrates the classifier, anomaly
  detection, knowledge retrieval and LLM generation into a single API.
  Exposes methods to train and load the classifier, fit anomaly stats,
  classify faults, detect anomalies and produce structured diagnostics.

Environment variables:

* ``OLLAMA_HOST`` (optional): URL where the Ollama service is running.
  Defaults to ``http://127.0.0.1:11434``.
* ``MODEL_NAME`` (optional): Name of the local model to use with
  Ollama.  Defaults to ``llama3.2:1b``.

The knowledge base JSON file should live next to this module at
``local_knowledge_base.json`` and contain a list of entries with
``title``, ``fault_type``, ``symptoms`` and ``diagnostic_steps`` fields.
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

# Base directory for locating resources relative to this file
BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2:1b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
CLASSIFIER_PATH = BASE_DIR / "artifacts" / "local_fault_classifier.joblib"
KNOWLEDGE_BASE_PATH = BASE_DIR / "local_knowledge_base.json"


class OllamaNotRunningError(RuntimeError):
    """Raised when the Ollama server cannot be reached."""


class LocalKnowledgeBase:
    """
    Simple TF‑IDF backed retriever over a JSON knowledge base.

    The JSON file is expected to contain a list of dicts with at least
    ``title``, ``fault_type``, ``symptoms`` and ``diagnostic_steps`` keys.
    """

    def __init__(self, kb_path: Path = KNOWLEDGE_BASE_PATH):
        if not kb_path.exists():
            raise FileNotFoundError(
                f"Knowledge base file '{kb_path}' is missing. Please create it before running diagnostics."
            )
        self.entries: List[Dict[str, str]] = json.loads(kb_path.read_text())
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
        """Return top_k knowledge base entries that match the query text."""
        if not text.strip():
            return self.entries[:top_k]
        query_vec = self.vectorizer.transform([text])
        scores = pairwise.cosine_similarity(query_vec, self.matrix)[0]
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [self.entries[idx] for idx in ranked_idx]


class OllamaClient:
    """
    Lightweight HTTP client for the local Ollama server.

    This class encapsulates interactions with the Ollama CLI and HTTP API
    without introducing a dependency on the ``ollama`` Python package,
    which may not be available on some systems.  The primary methods
    allow checking whether a model is installed and generating responses
    from a prompt.
    """

    def __init__(self, base_url: str = OLLAMA_HOST, model: str = MODEL_NAME):
        self.base_url = base_url.rstrip("/")
        self.model = model

    @staticmethod
    def _run_ollama_cli(args: Sequence[str]) -> subprocess.CompletedProcess:
        """Helper to invoke the ``ollama`` command line tool."""
        return subprocess.run(
            ["ollama", *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def ensure_model_available(self) -> str:
        """
        Ensure the desired model is available locally. Returns a human
        readable status message. If the model isn't installed, instructs
        the user how to install it.
        """
        try:
            result = self._run_ollama_cli(["list"])
        except FileNotFoundError as exc:
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
            # Keep the connect timeout short so the UI fails fast when the
            # Ollama server is down or unreachable. A longer read timeout
            # is allowed for generation to complete once the connection is
            # established.
            response = requests.post(url, json=payload, timeout=(5, 30))
        except requests.RequestException as exc:
            raise OllamaNotRunningError(
                f"Failed to reach Ollama at {self.base_url}. Ensure 'ollama serve' is running."
            ) from exc

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            error_detail = response.text if response.text else str(exc)
            raise OllamaNotRunningError(
                f"Ollama returned error: {response.status_code} - {error_detail}"
            ) from exc
        
        return response.json()

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Generate a response from the local LLM given a prompt.  Optionally
        include a system prompt to set behavior context.  Returns the
        plain text response.
        """
        payload: Dict[str, object] = {
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
    """
    Simple container for sensor readings used during diagnostics.  This
    class can be extended or replaced with an actual hardware interface
    if sensors are available on the Jetson.
    """

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
    """
    Brings together the classifier, anomaly detector, knowledge retriever
    and Ollama client into a single diagnostics agent.  This class
    exposes convenience methods to train and load the classifier,
    compute anomaly baselines, classify sensor snapshots, detect
    outliers and generate structured diagnostic reports using the
    local language model.
    """

    def __init__(
        self,
        knowledge_base: Optional[LocalKnowledgeBase] = None,
        ollama_client: Optional[OllamaClient] = None,
        classifier_path: Path = CLASSIFIER_PATH,
    ) -> None:
        # Lazy initialization of knowledge base and LLM client
        self.kb = knowledge_base or LocalKnowledgeBase()
        self.ollama = ollama_client or OllamaClient()
        self.classifier_path = classifier_path
        self.classifier: Optional[RandomForestClassifier] = None
        self.feature_columns: List[str] = []

        # Attributes for retrieval‑augmented generation (RAG)
        self.vector_store = None
        self.vector_store_dir = BASE_DIR / "vector_store"

    # ------------------------------------------------------------------
    # Fault classification helpers
    # ------------------------------------------------------------------
    def train_fault_classifier(self, data_path: Path) -> str:
        """Train a RandomForest classifier on a CSV with a 'fault_type' column."""
        df = pd.read_csv(data_path)
        if "fault_type" not in df.columns:
            raise ValueError("Dataset must contain a 'fault_type' column.")

        # Drop time column if present and select numeric columns
        X = df.drop(columns=["fault_type"], errors="ignore")
        if "Time" in X.columns:
            X = X.drop(columns=["Time"])
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols].fillna(X[numeric_cols].mean())
        y = df["fault_type"].astype(int)

        clf = RandomForestClassifier(n_estimators=300, random_state=7, n_jobs=-1)
        clf.fit(X, y)

        # Persist for later use
        self.classifier = clf
        self.feature_columns = numeric_cols
        self.classifier_path.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump({"model": clf, "features": numeric_cols}, self.classifier_path)

        return f"Trained classifier on {len(df)} samples with {len(numeric_cols)} features."

    def load_classifier(self) -> None:
        """Load the classifier from disk into memory."""
        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"Classifier file '{self.classifier_path}' not found. Run 'train_fault_classifier' first."
            )
        artifact = joblib.load(self.classifier_path)
        self.classifier = artifact["model"]
        self.feature_columns = artifact["features"]

    def classify_fault(self, sensor_row: Dict[str, float]) -> str:
        """Classify a single sensor snapshot into a fault type and return a string summary."""
        if self.classifier is None:
            self.load_classifier()

        row = pd.DataFrame([sensor_row])
        # Ensure we only use known feature columns
        row = row[self.feature_columns].fillna(0)
        prediction = int(self.classifier.predict(row)[0])
        proba = self.classifier.predict_proba(row)[0]
        confidence = float(np.max(proba))
        return f"Predicted fault_type={prediction} with confidence {confidence:.2f}"

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------
    def fit_anomaly_baseline(self, normal_data_path: Path) -> Dict[str, Dict[str, float]]:
        """
        Fit baseline mean/std statistics from a CSV of normal operation.
        Returns a dict of stats keyed by feature name.
        """
        df = pd.read_csv(normal_data_path)
        stats: Dict[str, Dict[str, float]] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std(ddof=0) or 1.0),
            }
        self.anomaly_stats = stats
        return stats

    def detect_anomaly(self, sensor_row: Dict[str, float], z_threshold: float = 3.0) -> Dict[str, float]:
        """
        Detect anomalies on a sensor snapshot using Z‑score.  Returns a
        dict of features with their Z‑scores if they exceed the given
        threshold.  ``fit_anomaly_baseline`` must be called beforehand.
        """
        if not hasattr(self, "anomaly_stats"):
            raise RuntimeError("Call 'fit_anomaly_baseline' before detecting anomalies.")

        anomalies: Dict[str, float] = {}
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
    # Retrieval‑augmented structured diagnostics
    # ------------------------------------------------------------------
    def generate_diagnostics(self, summary: str, sensor_row: Dict[str, float]) -> str:
        """
        Use the knowledge base, current sensor snapshot and a free‑text
        summary to produce a step‑by‑step diagnostic plan.  The output
        is formatted in markdown with three sections: Fault Hypotheses,
        Recommended Checks and Next Isolation Steps.  Requires the
        underlying Ollama server to be running.
        """
        retrievals = self.kb.query(summary, top_k=3)
        context = json.dumps(retrievals, indent=2)
        prompt = f"""
You are an embedded diagnostics expert operating offline on a Jetson Orin Nano.
Use ONLY the context below and the provided sensor snapshot to craft a
step‑by‑step diagnostic playbook. Output markdown with three sections:
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

    # ------------------------------------------------------------------
    # Retrieval‑Augmented Generation (RAG) utilities
    # ------------------------------------------------------------------
    def _ensure_vector_store_loaded(self) -> None:
        """Load a persisted vector store from disk if it exists and not already loaded."""
        if self.vector_store is not None:
            return
        if self.vector_store_dir.exists():
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_community.embeddings import HuggingFaceEmbeddings

                embeddings = HuggingFaceEmbeddings()
                self.vector_store = FAISS.load_local(str(self.vector_store_dir), embeddings)
            except Exception as exc:
                raise RuntimeError(f"Failed to load vector store: {exc}")

    def ingest_pdf(self, pdf_path: Path) -> str:
        """
        Ingest a datasheet PDF into the local vector store.  The PDF is
        split into text chunks, embedded using a HuggingFace model and
        indexed with FAISS.  If a vector store already exists, the
        new documents are added to it.  The index is persisted to
        ``vector_store_dir``.  Returns a human‑readable summary of
        ingestion results.
        """
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies for RAG PDF ingestion. Please install 'langchain', 'langchain-community', 'langchain-text-splitters', 'sentence-transformers' and 'faiss-cpu'."
            ) from exc

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file '{pdf_path}' not found.")

        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()

        if self.vector_store is None:
            if self.vector_store_dir.exists():
                try:
                    self._ensure_vector_store_loaded()
                except Exception:
                    self.vector_store = None
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(docs, embeddings)
            else:
                self.vector_store.add_documents(docs)
        else:
            self.vector_store.add_documents(docs)

        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.vector_store_dir))

        return f"Ingested {len(docs)} chunks from {pdf_path.name} into the vector store."

    def query_rag(self, question: str, top_k: int = 3) -> str:
        """
        Answer a user question using retrieval‑augmented generation.

        1. Ensure the vector store is loaded from disk.
        2. Perform a similarity search to retrieve the top_k relevant chunks.
        3. Compose a prompt combining the retrieved context and the question.
        4. Use the local LLM (Ollama) to generate a grounded answer.

        If no vector store exists, a user‑friendly message is returned.
        """
        self._ensure_vector_store_loaded()
        if self.vector_store is None:
            return "⚠️ No vector store found. Upload a datasheet PDF first."

        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            return "⚠️ Required RAG dependencies are not installed."

        docs_and_scores = self.vector_store.similarity_search(question, k=top_k)
        if not docs_and_scores:
            context = ""
        else:
            context = "\n\n".join([doc.page_content for doc in docs_and_scores])

        prompt = f"""
You are a technical assistant operating offline on a Jetson Orin Nano. You have access to
technical datasheets that describe components of a power electronics system. Use only the
following context from these datasheets to answer the question at the end. If the
question cannot be answered with the provided context, state that the information is
not available.

Context:
{context}

Question: {question}

Answer:"""
        return self.ollama.generate(prompt)
