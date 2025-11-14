"""
Main FastAPI server for iDAQ local diagnostics and training.

This server exposes a simple web interface for both administrators and
end users. Administrators (authenticated via static credentials) can
upload normal and fault datasets, trigger training of the fault
classifier and anomaly baseline, and view their status.  End users can
view live sensor readings plotted on a chart and interact with a
diagnostic chatbot powered by a locally running Ollama language model.

Endpoints:

* ``GET /`` – Redirects to login or appropriate dashboard based on cookie.
* ``GET /login`` – Render the login form.
* ``POST /login`` – Authenticate and set an admin cookie.
* ``GET /logout`` – Clear admin cookie and redirect to login.
* ``GET /admin`` – Upload and training interface for admins.
* ``POST /upload-normal`` – Upload a CSV of normal operation data.
* ``POST /upload-fault`` – Upload a CSV of fault data.
* ``POST /upload-pdf`` – Upload a datasheet PDF and ingest it for RAG.
* ``POST /train`` – Train the classifier and baseline using uploaded data.
* ``GET /user`` – User dashboard with live graph and chat.
* ``GET /sensor-data`` – Return current sensor values as JSON.
* ``POST /chat`` – Send a message to the local LLM and return its response.
* ``POST /ask-rag`` – Send a question through the RAG pipeline.

Note that this example uses a very simple cookie mechanism to manage
admin sessions. In a production system you would use a more robust
authentication framework.
"""

import os
import random
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from local_llama_agent import (
    LocalDiagnosticsAgent,
    OllamaNotRunningError,
)

# Base directory for data and templates
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

# Instantiate the diagnostics agent (lazy loads knowledge base and LLM)
agent = LocalDiagnosticsAgent()

# Create the FastAPI app
app = FastAPI(title="iDAQ Diagnostics Server")

# Ensure templates directory exists
if not TEMPLATES_DIR.exists():
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def read_template(name: str) -> str:
    """Read an HTML template from the templates directory."""
    path = TEMPLATES_DIR / name
    return path.read_text(encoding="utf-8")


def is_admin(request: Request) -> bool:
    """Check whether the incoming request has an admin cookie set."""
    return request.cookies.get("admin") == "1"


def redirect_to_login() -> RedirectResponse:
    return RedirectResponse(url="/login", status_code=303)


# ---------------------------------------------------------------------
# Authentication endpoints
# ---------------------------------------------------------------------

@app.get("/")
async def index(request: Request):
    """Root path; redirect users to appropriate dashboard."""
    if is_admin(request):
        return RedirectResponse(url="/admin", status_code=303)
    # If not admin and not authenticated, show user dashboard
    return RedirectResponse(url="/user", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_form():
    """Render the login form."""
    return HTMLResponse(read_template("login.html"))


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Authenticate user and set an admin cookie if credentials match."""
    # Replace these with secure credential storage in real usage
    if username == "pqlab" and password == "PQ2025!":
        response = RedirectResponse(url="/admin", status_code=303)
        # Set admin cookie; expires when browser session ends
        response.set_cookie(key="admin", value="1", httponly=True)
        return response
    else:
        # If authentication fails, redirect to user dashboard (no admin)
        return RedirectResponse(url="/user", status_code=303)


@app.get("/logout")
async def logout():
    """Clear the admin cookie and redirect to login page."""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key="admin")
    return response


# ---------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """Render the admin dashboard for uploading and training data."""
    if not is_admin(request):
        return redirect_to_login()
    return HTMLResponse(read_template("admin.html"))


@app.post("/upload-normal")
async def upload_normal(request: Request, file: UploadFile = File(...)):
    """
    Upload a CSV file containing normal operation data.  Saves the file
    as ``normal.csv`` in the app directory so that it can be used for
    anomaly baseline training.
    """
    if not is_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    contents = await file.read()
    data_path = BASE_DIR / "normal.csv"
    with open(data_path, "wb") as f:
        f.write(contents)
    return {"message": "Normal dataset uploaded."}


@app.post("/upload-pdf")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """
    Upload a datasheet PDF for retrieval‑augmented generation.  The PDF
    will be saved to a ``datasheets`` directory and ingested into the
    vector store via the diagnostics agent.  Only admins can perform
    this action.
    """
    if not is_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    contents = await file.read()
    datasheets_dir = BASE_DIR / "datasheets"
    datasheets_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = datasheets_dir / file.filename
    with open(pdf_path, "wb") as f:
        f.write(contents)
    try:
        msg = agent.ingest_pdf(pdf_path)
        return {"message": msg}
    except Exception as exc:
        return JSONResponse({"error": f"Error ingesting PDF: {exc}"}, status_code=500)


@app.post("/upload-fault")
async def upload_fault(request: Request, file: UploadFile = File(...)):
    """
    Upload a CSV file containing fault data.  Saves the file as
    ``fault.csv`` in the app directory so that it can be used for
    classifier training.
    """
    if not is_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    contents = await file.read()
    data_path = BASE_DIR / "fault.csv"
    with open(data_path, "wb") as f:
        f.write(contents)
    return {"message": "Fault dataset uploaded."}


@app.post("/train")
async def train_models(request: Request):
    """
    Trigger training of the fault classifier and anomaly detection baseline.
    Requires both ``fault.csv`` and ``normal.csv`` to exist.  Returns
    status messages detailing what was trained successfully.
    """
    if not is_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    messages = []
    fault_path = BASE_DIR / "fault.csv"
    normal_path = BASE_DIR / "normal.csv"
    if not fault_path.exists():
        messages.append("⚠️ fault.csv not found. Upload a fault dataset first.")
    else:
        try:
            msg = agent.train_fault_classifier(fault_path)
            messages.append(msg)
        except Exception as exc:
            messages.append(f"⚠️ Error training classifier: {exc}")
    if not normal_path.exists():
        messages.append("⚠️ normal.csv not found. Upload a normal dataset first.")
    else:
        try:
            stats = agent.fit_anomaly_baseline(normal_path)
            messages.append(
                f"✅ Fitted anomaly baseline on {normal_path.name} with {len(stats)} features."
            )
        except Exception as exc:
            messages.append(f"⚠️ Error fitting anomaly baseline: {exc}")
    return {"messages": messages}


# ---------------------------------------------------------------------
# User endpoints
# ---------------------------------------------------------------------

@app.get("/user", response_class=HTMLResponse)
async def user_page():
    """Render the user dashboard with live graph and chat interface."""
    return HTMLResponse(read_template("user.html"))


@app.get("/sensor-data")
async def sensor_data() -> dict:
    """
    Return simulated sensor data.  In a real deployment this function
    would interface with the Jetson GPIO or ADC hardware to read
    voltage, current, temperature and vibration signals.
    """
    return {
        "voltage": round(random.uniform(280.0, 320.0), 2),
        "current": round(random.uniform(10.0, 20.0), 2),
        "temperature": round(random.uniform(25.0, 80.0), 2),
        "vibration": round(random.uniform(0.0, 1.0), 2),
    }


@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    Accept a user message and return a response from the local LLM.  If
    the Ollama server is unreachable, returns an error message.
    """
    data = await request.json()
    message: Optional[str] = data.get("message")
    if not message:
        return JSONResponse({"response": "Please provide a message."}, status_code=400)
    try:
        response_text = agent.ollama.generate(message)
        return {"response": response_text}
    except OllamaNotRunningError as exc:
        return JSONResponse({"response": str(exc)}, status_code=503)


@app.post("/ask-rag")
async def ask_rag(request: Request):
    """
    Accept a question and return a response from the RAG pipeline.  The
    agent will retrieve relevant context from ingested datasheets and
    then query the local LLM to generate a grounded answer.
    """
    data = await request.json()
    question: Optional[str] = data.get("question") or data.get("message")
    if not question:
        return JSONResponse({"response": "Please provide a question."}, status_code=400)
    try:
        answer = agent.query_rag(question)
        return {"response": answer}
    except Exception as exc:
        return JSONResponse({"response": f"Error: {exc}"}, status_code=500)


@app.exception_handler(404)
async def not_found(request: Request, exc):
    """Handle 404 errors with a simple message."""
    return JSONResponse({"detail": "Not found"}, status_code=404)
