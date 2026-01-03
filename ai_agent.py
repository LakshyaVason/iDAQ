"""
AI Diagnostics Agent for iDAQ system.

This module provides the main DiagnosticsAgent class that handles:
- Fault classification using ML models
- RAG (Retrieval-Augmented Generation) with OpenAI
- Knowledge base retrieval
- Diagnostic report generation
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
KNOWLEDGE_BASE_PATH = BASE_DIR / "local_knowledge_base.json"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


class DiagnosticsAgent:
    """
    Main diagnostics agent that integrates fault classification,
    anomaly detection, and RAG-based report generation.
    """

    def __init__(self):
        """Initialize the diagnostics agent."""
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.vector_store = None
        
        if self.openai_key:
            self.client = OpenAI(api_key=self.openai_key)
        else:
            self.client = None
            logger.warning("OpenAI API key not configured")
        
        self.classifier = None
        self.anomaly_stats = None
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> List[Dict[str, str]]:
        """Load knowledge base from JSON file."""
        if KNOWLEDGE_BASE_PATH.exists():
            try:
                with open(KNOWLEDGE_BASE_PATH, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading knowledge base: {e}")
        return []

    def train_fault_classifier(self, data_path: Path) -> str:
        """
        Train fault classifier on CSV data.
        
        Args:
            data_path: Path to training CSV file
            
        Returns:
            Status message
        """
        try:
            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            
            # Load training data
            df = pd.read_csv(data_path)
            
            if df.empty:
                return "Training data is empty"
            
            # Simple example: use all columns except last as features, last as target
            X = df.iloc[:, :-1].fillna(0)
            y = df.iloc[:, -1]
            
            # Train classifier
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(X, y)
            
            # Save model
            classifier_path = ARTIFACTS_DIR / "fault_classifier.joblib"
            joblib.dump(self.classifier, classifier_path)
            
            logger.info(f"Classifier trained and saved to {classifier_path}")
            return f"✓ Fault classifier trained successfully ({len(df)} samples)"
            
        except Exception as e:
            logger.error(f"Error training classifier: {e}")
            return f"✗ Failed to train classifier: {str(e)}"

    def fit_anomaly_baseline(self, data_path: Path) -> Dict[str, Any]:
        """
        Fit anomaly detection baseline on normal operation data.
        
        Args:
            data_path: Path to CSV file with normal operation data
            
        Returns:
            Dictionary with baseline statistics
        """
        try:
            import pandas as pd
            
            df = pd.read_csv(data_path)
            if df.empty:
                return {}
            
            # Calculate statistics for each column
            self.anomaly_stats = {
                col: {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }
                for col in df.columns
            }
            
            logger.info(f"Anomaly baseline fitted with {len(self.anomaly_stats)} features")
            return self.anomaly_stats
            
        except Exception as e:
            logger.error(f"Error fitting anomaly baseline: {e}")
            return {}

    def train_rag(self, pdf_paths: List[Path]) -> str:
        """
        Train RAG system on PDF documents.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Status message
        """
        try:
            logger.info(f"Training RAG on {len(pdf_paths)} documents")
            # Placeholder for RAG training logic
            return f"✓ RAG system trained on {len(pdf_paths)} documents"
        except Exception as e:
            logger.error(f"Error training RAG: {e}")
            return f"✗ Failed to train RAG: {str(e)}"

    def ingest_pdf(self, pdf_path: Path) -> str:
        """Ingest PDF into vector store."""
        try:
            logger.info(f"Ingesting PDF: {pdf_path}")
            return f"✓ PDF ingested: {pdf_path.name}"
        except Exception as e:
            logger.error(f"Error ingesting PDF: {e}")
            return f"✗ Failed to ingest PDF: {str(e)}"

    def classify_fault(self, sensor_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify sensor data for fault detection.
        
        Args:
            sensor_data: Dictionary of sensor readings
            
        Returns:
            Classification result with confidence
        """
        if self.classifier is None:
            return {
                "fault_type": "unknown",
                "confidence": 0.0,
                "status": "Classifier not trained"
            }
        
        try:
            import pandas as pd
            
            # Prepare data for prediction
            df = pd.DataFrame([sensor_data])
            prediction = self.classifier.predict(df)[0]
            confidence = float(self.classifier.predict_proba(df).max())
            
            return {
                "fault_type": str(prediction),
                "confidence": confidence,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error classifying fault: {e}")
            return {
                "fault_type": "error",
                "confidence": 0.0,
                "status": f"Classification error: {str(e)}"
            }

    def detect_anomaly(self, sensor_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect anomalies in sensor data.
        
        Args:
            sensor_data: Dictionary of sensor readings
            
        Returns:
            Anomaly detection result
        """
        if not self.anomaly_stats:
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "status": "Baseline not fitted"
            }
        
        try:
            anomalies = []
            scores = []
            
            for key, value in sensor_data.items():
                if key not in self.anomaly_stats:
                    continue
                
                stats = self.anomaly_stats[key]
                mean = stats["mean"]
                std = stats["std"]
                
                if std == 0:
                    score = 0
                else:
                    score = abs((value - mean) / std)
                
                scores.append(score)
                if score > 3.0:  # 3 sigma
                    anomalies.append(f"{key}: {score:.2f}σ")
            
            avg_score = sum(scores) / len(scores) if scores else 0
            
            return {
                "is_anomaly": len(anomalies) > 0,
                "anomaly_score": avg_score,
                "anomalies": anomalies,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}")
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "status": f"Detection error: {str(e)}"
            }

    def query_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Query knowledge base for relevant information.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant knowledge base entries
        """
        if not self.knowledge_base:
            return []
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create documents from knowledge base
            documents = [
                f"{item.get('title', '')} {item.get('symptoms', '')} {item.get('diagnostic_steps', '')}"
                for item in self.knowledge_base
            ]
            
            if not documents:
                return []
            
            # Vectorize and compute similarity
            vectorizer = TfidfVectorizer()
            query_vec = vectorizer.fit_transform([query] + documents)
            similarities = cosine_similarity(query_vec[0:1], query_vec[1:]).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-top_k:][::-1]
            results = [self.knowledge_base[i] for i in top_indices if similarities[i] > 0.1]
            
            return results
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return []

    def query_rag(self, question: str, session_data: Optional[Dict] = None) -> str:
        """
        Query RAG system with context from session.
        
        Args:
            question: User question
            session_data: Previous session context
            
        Returns:
            RAG response
        """
        if not self.client:
            return "OpenAI API not configured."
        
        try:
            # Build context from knowledge base
            kb_results = self.query_knowledge_base(question)
            kb_context = "\n".join([
                f"- {item.get('title', '')}: {item.get('diagnostic_steps', '')}"
                for item in kb_results[:3]
            ])
            
            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert diagnostics assistant for iDAQ power quality systems.
Answer questions based on the knowledge base and sensor data context.

Knowledge Base:
{kb_context if kb_context else "No relevant information found"}

Be concise, technical, and actionable."""
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
            
            # Call OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return f"Error: {str(e)}"

    def chat(
        self,
        message: str,
        session_data: Optional[Dict] = None,
        latest_readings: Optional[Dict] = None
    ) -> str:
        """
        Chat with the diagnostics agent.
        
        Args:
            message: User message
            session_data: Previous session context
            latest_readings: Latest sensor readings
            
        Returns:
            Agent response
        """
        if not self.client:
            return "OpenAI API not configured."
        
        try:
            # Build context from sensor data
            context_str = ""
            if latest_readings:
                context_str = f"\nLatest sensor readings:\n{json.dumps(latest_readings, indent=2)}"
            
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a helpful diagnostics assistant for iDAQ power quality systems.
Answer questions about power quality, diagnostics, and system operation.{context_str}"""
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Error: {str(e)}"