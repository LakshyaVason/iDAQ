"""
AI Agent for iDAQ Diagnostics with OpenAI as primary reasoning engine.

This module provides:
- OpenAI GPT-4 for main chat and reasoning
- Ollama for local data compilation (simulation data → JSON)
- RAG with OpenAI embeddings and FAISS vector store
- Session management with Firebase
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier

# LangChain imports for RAG - with better error handling
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LangChain not fully available: {e}")
    LANGCHAIN_AVAILABLE = False

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
CLASSIFIER_PATH = BASE_DIR / "artifacts" / "fault_classifier.joblib"


@dataclass
class SensorSnapshot:
    """Container for sensor readings."""
    voltage: List[float]
    current: List[float]
    temperature: List[float]
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            "voltage": self.voltage,
            "current": self.current,
            "temperature": self.temperature,
            "timestamp": self.timestamp
        }


class DataCompiler:
    """Uses Ollama to compile simulation data into structured JSON."""
    
    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host
        self.model = "llama3.2:1b"
    
    def compile_sensor_data(self, raw_data: List[Dict]) -> Dict:
        """Compile raw sensor data into structured format for OpenAI."""
        try:
            import requests
            
            prompt = f"""
You are a data compiler. Convert this sensor data into a structured summary.

Raw Data:
{json.dumps(raw_data[-10:], indent=2)}

Return ONLY valid JSON with this structure:
{{
    "summary": "Brief summary of trends",
    "statistics": {{
        "voltage_avg": number,
        "current_avg": number,
        "temperature_avg": number,
        "voltage_trend": "rising/falling/stable",
        "anomalies": ["list of anomalies"]
    }},
    "recommendations": ["list of recommendations"]
}}
"""
            
            response = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=30
            )
            
            if response.ok:
                text = response.json().get("response", "")
                # Extract JSON from response
                try:
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start >= 0 and end > start:
                        return json.loads(text[start:end])
                except:
                    pass
            
            # Fallback: compute basic statistics
            return self._compute_basic_stats(raw_data)
            
        except Exception as e:
            print(f"Data compilation error: {e}")
            return self._compute_basic_stats(raw_data)
    
    def _compute_basic_stats(self, data: List[Dict]) -> Dict:
        """Fallback: compute basic statistics without LLM."""
        if not data:
            return {"summary": "No data available", "statistics": {}, "recommendations": []}
        
        voltages = [d["voltage"][0] for d in data if "voltage" in d]
        currents = [d["current"][0] for d in data if "current" in d]
        temps = [d["temperature"][0] for d in data if "temperature" in d]
        
        return {
            "summary": f"Analyzed {len(data)} data points",
            "statistics": {
                "voltage_avg": np.mean(voltages) if voltages else 0,
                "current_avg": np.mean(currents) if currents else 0,
                "temperature_avg": np.mean(temps) if temps else 0,
                "voltage_trend": "stable",
                "anomalies": []
            },
            "recommendations": ["Continue monitoring"]
        }


class DiagnosticsAgent:
    """Main AI agent using OpenAI for reasoning and RAG."""
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # RAG components
        self.vector_store_dir = BASE_DIR / "vector_store"
        self.vector_store: Optional[FAISS] = None
        self.embeddings = None
        
        if LANGCHAIN_AVAILABLE:
            try:
                self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                self._load_vector_store()
            except Exception as e:
                print(f"Warning: Could not initialize embeddings: {e}")
        
        self.data_compiler = DataCompiler()
        
        self.classifier: Optional[RandomForestClassifier] = None
        self.feature_columns: List[str] = []
        self.anomaly_stats: Dict = {}
    
    # ===== Classifier Methods =====
    
    def train_fault_classifier(self, data_path: Path) -> str:
        """Train RandomForest classifier on fault data."""
        df = pd.read_csv(data_path)
        if "fault_type" not in df.columns:
            raise ValueError("Dataset must contain 'fault_type' column")
        
        X = df.drop(columns=["fault_type", "Time"], errors="ignore")
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols].fillna(X[numeric_cols].mean())
        y = df["fault_type"].astype(int)
        
        clf = RandomForestClassifier(n_estimators=300, random_state=7, n_jobs=-1)
        clf.fit(X, y)
        
        self.classifier = clf
        self.feature_columns = numeric_cols
        CLASSIFIER_PATH.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump({"model": clf, "features": numeric_cols}, CLASSIFIER_PATH)
        
        return f"✅ Trained classifier on {len(df)} samples with {len(numeric_cols)} features"
    
    def load_classifier(self) -> None:
        """Load classifier from disk."""
        if not CLASSIFIER_PATH.exists():
            raise FileNotFoundError("Classifier not found. Train first.")
        
        artifact = joblib.load(CLASSIFIER_PATH)
        self.classifier = artifact["model"]
        self.feature_columns = artifact["features"]
    
    def classify_fault(self, sensor_row: Dict) -> str:
        """Classify fault type from sensor reading."""
        if self.classifier is None:
            self.load_classifier()
        
        row = pd.DataFrame([sensor_row])[self.feature_columns].fillna(0)
        prediction = int(self.classifier.predict(row)[0])
        proba = self.classifier.predict_proba(row)[0]
        confidence = float(np.max(proba))
        
        return f"Fault Type {prediction} (confidence: {confidence:.2%})"
    
    # ===== Anomaly Detection =====
    
    def fit_anomaly_baseline(self, normal_data_path: Path) -> Dict:
        """Fit baseline statistics from normal operation data."""
        df = pd.read_csv(normal_data_path)
        stats = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std(ddof=0) or 1.0)
            }
        
        self.anomaly_stats = stats
        return stats
    
    def detect_anomaly(self, sensor_row: Dict, z_threshold: float = 3.0) -> Dict:
        """Detect anomalies using Z-score."""
        if not self.anomaly_stats:
            raise RuntimeError("Call fit_anomaly_baseline first")
        
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
    
    # ===== RAG Methods =====
    
    def _load_vector_store(self) -> None:
        """Load existing vector store if available."""
        if not LANGCHAIN_AVAILABLE or not self.embeddings:
            print("⚠️ LangChain not available - RAG disabled")
            return
            
        if self.vector_store_dir.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(self.vector_store_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ Vector store loaded successfully")
            except Exception as e:
                print(f"⚠️ Could not load vector store: {e}")
    
    def ingest_pdf(self, pdf_path: Path) -> str:
        """Ingest PDF into vector store."""
        if not LANGCHAIN_AVAILABLE:
            return "❌ LangChain dependencies not installed. Run: pip install langchain langchain-openai langchain-community"
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = splitter.split_documents(documents)
            
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(docs, self.embeddings)
            else:
                self.vector_store.add_documents(docs)
            
            self.vector_store_dir.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(self.vector_store_dir))
            
            return f"✅ Ingested {len(docs)} chunks from {pdf_path.name}"
        
        except Exception as e:
            return f"❌ PDF ingestion failed: {str(e)}"
    
    def query_rag(self, question: str, session_data: Optional[List[Dict]] = None) -> str:
        """Answer question using RAG with sensor context."""
        print(f"[RAG] Question: {question}")
        print(f"[RAG] Vector store exists: {self.vector_store is not None}")
        
        if not LANGCHAIN_AVAILABLE:
            return "❌ RAG not available. Install dependencies: pip install langchain langchain-openai langchain-community faiss-cpu"
        
        if self.vector_store is None:
            return "⚠️ No datasheets uploaded yet. Please upload a PDF first in RAG mode."
        
        try:
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(question, k=4)
            print(f"[RAG] Retrieved {len(docs)} documents")
            
            if not docs:
                return "⚠️ No relevant information found in uploaded datasheets."
            
            # Build context from retrieved docs
            context = "\n\n".join([f"[Document {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)])
            
            # Add sensor context if available
            sensor_context = ""
            if session_data:
                compiled = self.data_compiler.compile_sensor_data(session_data)
                sensor_context = f"\n\nCurrent System Status:\n{json.dumps(compiled, indent=2)}"
            
            # Query OpenAI with context
            prompt = f"""You are a technical assistant with access to power electronics datasheets. 
Use ONLY the following context from datasheets to answer the question. 
If the answer is not in the context, say so clearly.

Context from Datasheets:
{context}
{sensor_context}

Question: {question}

Answer (be specific and cite information from the datasheets):"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a technical documentation expert. Answer based only on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            answer = response.choices[0].message.content
            print(f"[RAG] Response generated: {len(answer)} chars")
            return answer
        
        except Exception as e:
            error_msg = f"❌ RAG query error: {str(e)}"
            print(f"[RAG] Error: {e}")
            return error_msg
    
    # ===== Main Chat Method =====
    
    def chat(
        self,
        message: str,
        session_data: Optional[List[Dict]] = None,
        latest_readings: Optional[Dict] = None
    ) -> str:
        """Main chat interface using OpenAI GPT-4."""
        try:
            # Build context
            context_parts = []
            
            if latest_readings:
                context_parts.append("Latest Sensor Readings:")
                context_parts.append(json.dumps(latest_readings, indent=2))
            
            if session_data and len(session_data) > 0:
                compiled = self.data_compiler.compile_sensor_data(session_data)
                context_parts.append("\nSession Data Analysis:")
                context_parts.append(json.dumps(compiled, indent=2))
                
                # Add recent data points for time-based queries
                context_parts.append("\nRecent Data Points (last 10):")
                context_parts.append(json.dumps(session_data[-10:], indent=2))
            
            system_prompt = """You are an expert power electronics diagnostics assistant for an iDAQ monitoring system. You help users:
- Interpret sensor readings (voltage, current, temperature)
- Diagnose faults in inverters, converters, and power electronics
- Explain trends and anomalies
- Provide troubleshooting recommendations

Be concise, technical, and actionable. Always reference specific sensor values when discussing data."""

            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            if context_parts:
                messages.append({
                    "role": "system",
                    "content": "\n".join(context_parts)
                })
            
            messages.append({"role": "user", "content": message})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"❌ Chat error: {str(e)}"
    
    def generate_diagnostics(
        self,
        summary: str,
        sensor_row: Dict,
        session_data: Optional[List[Dict]] = None
    ) -> str:
        """Generate comprehensive diagnostic report."""
        compiled = {}
        if session_data:
            compiled = self.data_compiler.compile_sensor_data(session_data)
        
        prompt = f"""Generate a diagnostic report for this power electronics system.

Current Reading:
{json.dumps(sensor_row, indent=2)}

Session Analysis:
{json.dumps(compiled, indent=2)}

Issue Summary: {summary}

Provide:
1. **Fault Hypotheses** - What could be wrong?
2. **Recommended Checks** - Step-by-step testing procedure
3. **Next Isolation Steps** - How to narrow down the root cause

Format as markdown."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a power electronics diagnostics expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Diagnostics generation error: {str(e)}"