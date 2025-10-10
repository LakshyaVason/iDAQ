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
  for conversational queries about the data and results.  The LLM is loaded
  using an environment variable ``HF_TOKEN`` and will return concise, helpful
  answers.  If the model cannot be loaded, the chat interface is disabled.

Instructions
------------
Before running this app you should:

1. Install the required Python packages:

   ```bash
   pip install streamlit pandas scikit-learn python-dotenv transformers
   ```

   Installing ``transformers`` also pulls in PyTorch.  To run large models
   locally you will need a GPU with sufficient memory; otherwise the model
   loading will fail and the chat interface will be disabled.

2. Define an environment variable ``HF_TOKEN`` containing your Hugging Face
   access token.  You can set this in a ``.env`` file in the same directory
   as this script, or export it in your shell.  The token must have read
   permissions for the chosen model (for example
   ``meta-llama/Meta-Llama-3.1-8B-Instruct``).

3. Obtain the fault data CSV files manually (e.g. from the GPVS‑Faults or
   four‑leg inverter datasets) and save them on your computer.  Each file
   should be named such that the first two characters encode the fault type
   (``F0`` = healthy, ``F1``‑``F7`` = fault types).  Upload these files via
   the Streamlit interface when prompted.

Once the app is running, upload your training and testing files, click the
“Train Model” button and wait for the training to complete.  The
classification report will appear in the main panel.  Then you can chat
with the LLM using the text input field to ask questions about the data.
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

# Try to import transformers.  It is optional – if unavailable the chat
# interface will be disabled.
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def load_and_label(files: List[Any]) -> pd.DataFrame:
    """Load one or more CSV files and assign a fault_type label based on the
    file name.  Each row retains all numeric sensor columns, and the time
    column is kept for reference but excluded from the model.

    Parameters
    ----------
    files : list of Streamlit UploadedFile objects
        The CSV files uploaded by the user.  The first two characters of the
        file name should be ``F0``–``F7`` indicating the fault type.

    Returns
    -------
    pd.DataFrame
        Combined data frame containing all rows from the uploaded files with
        an additional ``fault_type`` column indicating the fault class.
    """
    data_frames = []
    for f in files:
        # Read the CSV into a DataFrame.  Allow thousands separators and
        # ignore potential malformed lines.
        df = pd.read_csv(f)
        # Determine fault type from file name.  We use a regular expression
        # matching F followed by a digit.
        match = re.match(r"F(\d)", f.name)
        fault_num = 0
        if match:
            fault_num = int(match.group(1))
        df["fault_type"] = fault_num
        data_frames.append(df)
    if data_frames:
        full_df = pd.concat(data_frames, ignore_index=True)
        return full_df
    else:
        return pd.DataFrame()


def train_random_forest(df: pd.DataFrame) -> Tuple[RandomForestClassifier, dict]:
    """Train a RandomForestClassifier to predict fault_type.

    The function drops the ``Time`` column if present, splits the data into
    training and validation sets (70/30), and trains a random forest.  It
    returns the fitted model and a classification report computed on the
    validation set.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing numeric feature columns and a ``fault_type``
        target column.

    Returns
    -------
    (RandomForestClassifier, dict)
        A tuple containing the fitted classifier and a classification report
        dictionary as produced by ``sklearn.metrics.classification_report``.
    """
    # Separate features and target
    X = df.drop(columns=["fault_type"])
    if "Time" in X.columns:
        X = X.drop(columns=["Time"])
    y = df["fault_type"].astype(int)

    # Handle any non-numeric columns that might be present (drop them)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    # Fill NaNs with column means
    X = X.fillna(X.mean())

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train a random forest classifier
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    return clf, report


def generate_classification_report(report: dict) -> str:
    """Format the classification report dictionary into a readable string.

    Parameters
    ----------
    report : dict
        The classification report as returned by ``sklearn``.

    Returns
    -------
    str
        A formatted string summarising precision, recall, f1‑score and support
        for each class as well as overall accuracy.
    """
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
    # Add accuracy
    acc = report.get("accuracy", 0)
    lines.append("\nOverall accuracy: {:.2f}".format(acc))
    return "\n".join(lines)


def load_llm() -> tuple:
    """Attempt to load a text generation pipeline for chatting.

    Reads the Hugging Face access token from the environment variable
    ``HF_TOKEN`` and attempts to construct a pipeline using the specified
    model.  If ``transformers`` is not installed or the model cannot be
    loaded, the function returns ``(None, None)``.

    Returns
    -------
    (pipeline, str)
        A tuple containing the pipeline object (or ``None`` if not
        available) and the model identifier used.  The pipeline can be used
        to generate responses in the chat interface.
    """
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    # Load environment variables.  Do this lazily to avoid requiring
    # python-dotenv when not needed.  If python-dotenv is unavailable the
    # environment variables must already be set in the system environment.
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        return None, None
    # Choose a model.  Meta's Llama 3 8B Instruct is used here; you may
    # substitute a different model accessible with your token.  Smaller
    # models (e.g. ``HuggingFaceH4/zephyr-7b-beta``) may load faster on
    # CPU‑only systems.
    model_id = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    try:
        llm_pipeline = pipeline(
            "text-generation",
            model=model_id,
            token=hf_token,
            device = 0,
            # Using bfloat16 reduces memory footprint on compatible GPUs.  On
            # CPU this argument may be ignored.
            dtype=torch.bfloat16 if hasattr(torch, "bfloat16") else None,
        )
        return llm_pipeline, model_id
    except Exception:
        # If loading fails (e.g. due to missing model card or GPU), return None.
        return None, None


def prepare_prompt(messages: List[dict]) -> str:
    """Assemble a conversation history into a single prompt for the LLM.

    The chat messages are converted into a simple structured format that
    distinguishes system, user and assistant turns.  This format loosely
    follows the one used by some open source chat models.

    Parameters
    ----------
    messages : list of dict
        Each message dict has ``role`` ("system", "user" or "assistant") and
        ``content`` (string) keys.

    Returns
    -------
    str
        A prompt string that concatenates the messages with role tags.
    """
    prompt_parts = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            prompt_parts.append(f"[SYSTEM] {content}\n")
        elif role == "user":
            prompt_parts.append(f"[USER] {content}\n")
        elif role == "assistant":
            prompt_parts.append(f"[ASSISTANT] {content}\n")
    return "".join(prompt_parts)


def app() -> None:
    """Run the Streamlit application."""
    # Import streamlit lazily so that the rest of the module can be used
    # without requiring streamlit at import time.  This avoids errors
    # when running tests or scripts that do not install streamlit.
    import streamlit as st  # type: ignore

    st.set_page_config(page_title="Power Electronics Fault Detection", layout="wide")
    st.title("🔌 Power Electronics Fault Detection Tool")

    st.markdown(
        """
        This tool helps you train a simple machine learning model to detect
        electrical faults in power electronics data (e.g. from a totem‑pole
        converter, inverter or DC‑DC converter) and interact with an
        optional large language model (LLM) to ask questions about your data.
        """
    )

    # Sidebar: dataset upload
    st.sidebar.header("Dataset upload")
    st.sidebar.markdown(
        "Please upload one or more CSV files for training and testing.  Each file's name\n"
        "must begin with 'F0'–'F7' indicating the fault scenario.  For example, 'F3M.csv'\n"
        "should contain data for fault type 3 in MPPT mode."
    )
    train_files = st.sidebar.file_uploader(
        "Training CSV files", accept_multiple_files=True, type=["csv"]
    )
    test_files = st.sidebar.file_uploader(
        "Testing CSV files", accept_multiple_files=True, type=["csv"]
    )

    # Load training data if available
    train_df = pd.DataFrame()
    if train_files:
        train_df = load_and_label(train_files)
        st.sidebar.success(f"Loaded {len(train_df)} training rows from {len(train_files)} file(s).")

    # Load testing data if available
    test_df = pd.DataFrame()
    if test_files:
        test_df = load_and_label(test_files)
        st.sidebar.success(f"Loaded {len(test_df)} testing rows from {len(test_files)} file(s).")

    # Display dataset summary in the main area
    if not train_df.empty:
        st.subheader("Training Data Summary")
        fault_counts = train_df["fault_type"].value_counts().sort_index()
        fault_summary_df = pd.DataFrame(
            {
                "Fault Type": fault_counts.index,
                "Count": fault_counts.values,
            }
        )
        st.table(fault_summary_df)
    else:
        st.info(
            "Upload training files to see a summary and train the model."
        )

    # Train model button
    model = None
    report = None
    if not train_df.empty:
        if st.button("Train Model"):
            with st.spinner("Training the random forest model…"):
                model, report = train_random_forest(train_df)
            st.success("Model training complete!")
            st.subheader("Validation Metrics")
            report_str = generate_classification_report(report)
            st.text(report_str)

    # Apply model to test data if available
    if model is not None and not test_df.empty:
        if st.button("Evaluate on Test Data"):
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
                # Show confusion matrix as a dataframe
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"True F{i}" for i in range(cm.shape[0])],
                    columns=[f"Pred F{i}" for i in range(cm.shape[1])],
                )
                st.subheader("Confusion Matrix")
                st.table(cm_df)

    # Chat interface section
    st.subheader("📢 Chat with the LLM")
    st.markdown(
        "Ask questions about your dataset, fault detection results or general power electronics.\n"
        "The chat will use a local LLM if available.  Messages are not stored outside the current session."
    )
    # Initialize chat history in session_state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert electrical engineer and data scientist.  "
                    "Answer questions concisely about power electronics, fault detection and the user's data."
                ),
            }
        ]

    # Load LLM pipeline once and cache it in the session state
    if "llm" not in st.session_state:
        llm_pipeline, model_id = load_llm()
        st.session_state.llm = llm_pipeline
        st.session_state.llm_id = model_id

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.chat_message("assistant").markdown(msg["content"])
        elif msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask a question…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Prepare prompt and call LLM if available
        llm = st.session_state.llm
        if llm is not None:
            with st.spinner("Generating response…"):
                prompt = prepare_prompt(st.session_state.messages)
                try:
                    # Generate up to 256 tokens in the response
                    response = llm(
                        prompt,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.95,
                    )
                    generated_text = response[0].get("generated_text", "")
                    # Extract the assistant's reply by removing the prompt from the output
                    assistant_reply = generated_text[len(prompt):].strip()
                except Exception as exc:
                    assistant_reply = (
                        "I couldn't generate a response due to an error.  "
                        "Please check your model configuration."
                    )
        else:
            assistant_reply = (
                "The LLM is not available.  Please set HF_TOKEN and install ``transformers`` "
                "to enable the chat feature."
            )
        # Append assistant reply to the conversation
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        st.chat_message("assistant").markdown(assistant_reply)


if __name__ == "__main__":
    app()