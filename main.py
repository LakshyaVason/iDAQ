"""
Streamlit application for training a fault detection model on power electronics data
and interacting with a local large language model (LLM) to ask questions about
the data.  The app is designed for use with datasets similar to the
Grid‑connected PV System Faults (GPVS‑Faults) dataset, which contain time
series measurements of voltage, current and other electrical signals under
various operating conditions and faults.  Users can manually upload CSV files
for training and testing, build a simple classification model to detect fault
types, and use an LLM chat interface to explore the data and results.

Features
--------
* Upload one or more training CSV files and one or more testing CSV files.
* Automatically assign fault labels based on file names (e.g. files starting
  with ``F1`` are labelled fault type 1).  A zero fault type denotes normal
  operation.
* Train a scikit‑learn ``RandomForestClassifier`` on the numeric features
  (dropping the ``Time`` column).  The trained model predicts the fault
  category for each row in the test dataset.
* Display a classification report and confusion matrix to evaluate model
  performance.
* Summarise the distribution of fault types in the training dataset.
* Provide a chat interface powered by a Hugging Face LLM (e.g. Meta Llama 3)
  for conversational queries about the data and results, using LangChain.
"""

import os
import re
from typing import List, Tuple, Any

import torch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Try to import transformers. It is optional – if unavailable the chat
# interface will be disabled.
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import LangChain components for LLM support
try:
    from langchain.llms import HuggingFacePipeline
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

def load_and_label(files: List[Any]) -> pd.DataFrame:
    """Load one or more CSV files and assign a fault_type label based on the file name."""
    data_frames = []
    for f in files:
        df = pd.read_csv(f)
        match = re.match(r"F(\d)", f.name)
        fault_num = int(match.group(1)) if match else 0
        df["fault_type"] = fault_num
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def train_random_forest(df: pd.DataFrame) -> Tuple[RandomForestClassifier, dict]:
    """Train a RandomForestClassifier to predict fault_type."""
    X = df.drop(columns=["fault_type"])
    if "Time" in X.columns:
        X = X.drop(columns=["Time"])
    y = df["fault_type"].astype(int)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols].fillna(X.mean())
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    return clf, report

def generate_classification_report(report: dict) -> str:
    """Format the classification report dictionary into a readable string."""
    lines = []
    header = f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1‑Score':<10} {'Support':<10}"
    lines.append(header)
    lines.append("-" * len(header))
    for cls, metrics in report.items():
        if cls in {"accuracy", "macro avg", "weighted avg"}:
            continue
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        f1 = metrics.get("f1-score", 0)
        support = metrics.get("support", 0)
        lines.append(f"{cls:<10} {precision:<10.2f} {recall:<10.2f} {f1:<10.2f} {support:<10}")
    acc = report.get("accuracy", 0)
    lines.append("\nOverall accuracy: {:.2f}".format(acc))
    return "\n".join(lines)

def load_llm_chain():
    """Load a HuggingFace model via LangChain and wrap it in a ConversationChain."""
    if not TRANSFORMERS_AVAILABLE or not LANGCHAIN_AVAILABLE:
        return None, None
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        return None, None
    model_id = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    try:
        hf_pipeline = pipeline(
            "text-generation",
            model=model_id,
            token=hf_token,
            device=0,
            dtype=torch.bfloat16 if hasattr(torch, "bfloat16") else None,
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        memory = ConversationBufferMemory()
        chain = ConversationChain(llm=llm, memory=memory)
        return chain, model_id
    except Exception:
        return None, None

def app() -> None:
    """Run the Streamlit application."""
    import streamlit as st  # type: ignore

    st.set_page_config(page_title="Power Electronics Fault Detection", layout="wide")
    st.title("🔌 Power Electronics Fault Detection Tool")

    st.markdown(
        """
        This tool helps you train a simple machine learning model to detect
        electrical faults in power electronics data and interact with an
        optional large language model (LLM) to ask questions about your data.
        """
    )

    st.sidebar.header("Dataset upload")
    st.sidebar.markdown(
        "Upload CSV files for training and testing. Each file's name must begin with 'F0'–'F7' indicating the fault type."
    )
    train_files = st.sidebar.file_uploader(
        "Training CSV files", accept_multiple_files=True, type=["csv"]
    )
    test_files = st.sidebar.file_uploader(
        "Testing CSV files", accept_multiple_files=True, type=["csv"]
    )

    train_df = load_and_label(train_files) if train_files else pd.DataFrame()
    test_df  = load_and_label(test_files)  if test_files  else pd.DataFrame()

    if not train_df.empty:
        st.subheader("Training Data Summary")
        fault_counts = train_df["fault_type"].value_counts().sort_index()
        fault_summary_df = pd.DataFrame({
            "Fault Type": fault_counts.index,
            "Count": fault_counts.values,
        })
        st.table(fault_summary_df)
    else:
        st.info("Upload training files to see a summary and train the model.")

    model = None
    report = None
    if not train_df.empty and st.button("Train Model"):
        with st.spinner("Training the random forest model…"):
            model, report = train_random_forest(train_df)
        st.success("Model training complete!")
        st.subheader("Validation Metrics")
        report_str = generate_classification_report(report)
        st.text(report_str)

    if model is not None and not test_df.empty and st.button("Evaluate on Test Data"):
        with st.spinner("Evaluating on test set…"):
            X_test = test_df.drop(columns=["fault_type"], errors="ignore")
            if "Time" in X_test.columns:
                X_test = X_test.drop(columns=["Time"])
            numeric_cols = X_test.select_dtypes(include=[np.number]).columns
            X_test = X_test[numeric_cols].fillna(X_test.mean())
            y_test = test_df["fault_type"].astype(int)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Test accuracy: {acc:.2f}")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=[f"True F{i}" for i in range(cm.shape[0])],
                columns=[f"Pred F{i}" for i in range(cm.shape[1])],
            )
            st.subheader("Confusion Matrix")
            st.table(cm_df)

    st.subheader("📢 Chat with the LLM")
    st.markdown(
        "Ask questions about your dataset, fault detection results or general power electronics.\n"
        "This uses a LangChain `ConversationChain` if a suitable LLM is available."
    )
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert electrical engineer and data scientist. "
                    "Answer questions concisely about power electronics, fault detection and the user's data."
                ),
            }
        ]

    if "llm_chain" not in st.session_state:
        chain, model_id = load_llm_chain()
        st.session_state.llm_chain = chain
        st.session_state.llm_model_id = model_id

    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.chat_message("assistant").markdown(msg["content"])
        elif msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])

    user_input = st.chat_input("Ask a question…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        chain = st.session_state.llm_chain
        if chain is not None:
            try:
                with st.spinner("Generating response…"):
                    assistant_reply = chain.predict(input=user_input)
            except Exception:
                assistant_reply = (
                    "I couldn't generate a response due to an error. "
                    "Please check your model configuration."
                )
        else:
            assistant_reply = (
                "The LLM is not available. Please set HF_TOKEN and install the required packages."
            )
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        st.chat_message("assistant").markdown(assistant_reply)

if __name__ == "__main__":
    app()
