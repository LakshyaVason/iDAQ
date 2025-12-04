"""
Complete FastAPI server for iDAQ diagnostics with Firebase authentication.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from local_llama_agent import (
    LocalDiagnosticsAgent,
    OllamaNotRunningError,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for data and templates
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DATASHEETS_DIR = BASE_DIR / "datasheets"

load_dotenv()

# Ensure directories exist
DATASHEETS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Firebase configuration
REQUIRED_FIREBASE_CONFIG_KEYS = [
    "apiKey",
    "authDomain",
    "projectId",
    "storageBucket",
    "messagingSenderId",
    "appId",
]

firebase_app: Optional[firebase_admin.App] = None

# Instantiate the diagnostics agent
agent = LocalDiagnosticsAgent()

# Create the FastAPI app
app = FastAPI(title="iDAQ Diagnostics Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def read_template(name: str) -> str:
    """Read an HTML template from the templates directory."""
    path = TEMPLATES_DIR / name
    if not path.exists():
        logger.error(f"Template {name} not found at {path}")
        return f"<html><body><h1>Template {name} not found</h1></body></html>"
    return path.read_text(encoding="utf-8")


def is_admin(request: Request) -> bool:
    """Check whether the incoming request has an admin cookie set."""
    return request.cookies.get("admin") == "1"


def redirect_to_login() -> RedirectResponse:
    return RedirectResponse(url="/login", status_code=303)


def _load_service_account_credentials() -> credentials.Base:
    """Load Firebase service account credentials from environment variables."""
    key_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_FILE")
    key_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
    project_id = os.environ.get("FIREBASE_PROJECT_ID")

    if key_json:
        try:
            parsed = json.loads(key_json.replace("\\n", "\n"))
            return credentials.Certificate(parsed)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid FIREBASE_SERVICE_ACCOUNT JSON: {exc}")

    if key_path and Path(key_path).exists():
        return credentials.Certificate(key_path)

    if project_id:
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
    return credentials.ApplicationDefault()


def init_firebase_admin():
    """Initialize the Firebase Admin SDK if it has not been initialized."""
    global firebase_app
    if firebase_admin._apps:
        firebase_app = firebase_admin.get_app()
        return firebase_app

    cred = _load_service_account_credentials()
    firebase_app = firebase_admin.initialize_app(cred)
    return firebase_app


def get_firebase_client_config() -> dict:
    """Assemble Firebase client configuration from environment variables."""
    config = {
        "apiKey": os.environ.get("FIREBASE_API_KEY"),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN"),
        "projectId": os.environ.get("FIREBASE_PROJECT_ID"),
        "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": os.environ.get("FIREBASE_APP_ID"),
    }

    measurement_id = os.environ.get("FIREBASE_MEASUREMENT_ID")
    if measurement_id:
        config["measurementId"] = measurement_id

    missing_keys = [k for k in REQUIRED_FIREBASE_CONFIG_KEYS if not config.get(k)]
    if missing_keys:
        raise RuntimeError(
            "Missing Firebase client configuration keys: " + ", ".join(missing_keys)
        )

    return config


async def verify_firebase_token(authorization: str = Header(None)) -> dict:
    """Verify a Firebase ID token from the Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = authorization.split(" ", 1)[1]
    init_firebase_admin()
    try:
        decoded = firebase_auth.verify_id_token(token)
        return decoded
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid ID token: {exc}")


# ---------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Check if Ollama is running and Firebase is configured on startup."""
    logger.info("=" * 60)
    logger.info("iDAQ Diagnostics Server Starting Up")
    logger.info("=" * 60)
    
    try:
        init_firebase_admin()
        logger.info("✅ Firebase Admin SDK initialized")
    except Exception as exc:
        logger.warning(f"⚠️ WARNING: Firebase Admin initialization failed: {exc}")
        logger.warning("Firebase features will be disabled")
    
    try:
        status = agent.ollama.ensure_model_available()
        logger.info(status)
    except Exception as e:
        logger.warning(f"⚠️ WARNING: {e}")
        logger.warning("Make sure to:")
        logger.warning("  1. Install Ollama: https://ollama.com")
        logger.warning("  2. Start Ollama service: 'ollama serve'")
        logger.warning("  3. Pull the model: 'ollama pull llama3.2:1b'")
    
    logger.info("=" * 60)


# ---------------------------------------------------------------------
# Authentication endpoints
# ---------------------------------------------------------------------

@app.get("/")
async def index(request: Request):
    """Root path; redirect users to appropriate dashboard."""
    if is_admin(request):
        return RedirectResponse(url="/admin", status_code=303)
    return RedirectResponse(url="/user", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_form():
    """Render the login form."""
    return HTMLResponse(read_template("login.html"))


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Authenticate user and set an admin cookie if credentials match."""
    if username == "pqlab" and password == "PQ2025!":
        response = RedirectResponse(url="/admin", status_code=303)
        response.set_cookie(key="admin", value="1", httponly=True)
        return response
    else:
        return RedirectResponse(url="/user", status_code=303)


@app.get("/logout")
async def logout():
    """Clear the admin cookie and redirect to login page."""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key="admin")
    return response


@app.get("/config/firebase")
async def firebase_config():
    """Expose Firebase client configuration to the frontend."""
    try:
        return get_firebase_client_config()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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
    """Upload a CSV file containing normal operation data."""
    if not is_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    
    try:
        contents = await file.read()
        data_path = BASE_DIR / "normal.csv"
        with open(data_path, "wb") as f:
            f.write(contents)
        logger.info(f"Normal dataset uploaded: {len(contents)} bytes")
        return {"message": "Normal dataset uploaded successfully."}
    except Exception as e:
        logger.error(f"Error uploading normal dataset: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/upload-fault")
async def upload_fault(request: Request, file: UploadFile = File(...)):
    """Upload a CSV file containing fault data."""
    if not is_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    
    try:
        contents = await file.read()
        data_path = BASE_DIR / "fault.csv"
        with open(data_path, "wb") as f:
            f.write(contents)
        logger.info(f"Fault dataset uploaded: {len(contents)} bytes")
        return {"message": "Fault dataset uploaded successfully."}
    except Exception as e:
        logger.error(f"Error uploading fault dataset: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    user: dict = Depends(verify_firebase_token),
):
    """Upload a datasheet PDF for RAG. Requires authentication."""
    try:
        if not file.filename.endswith('.pdf'):
            logger.warning(f"Invalid file type attempted: {file.filename}")
            return JSONResponse(
                {"error": "Only PDF files are allowed"}, 
                status_code=400
            )
        
        contents = await file.read()
        
        if len(contents) == 0:
            logger.error("Empty file uploaded")
            return JSONResponse(
                {"error": "Uploaded file is empty"}, 
                status_code=400
            )
        
        pdf_path = DATASHEETS_DIR / file.filename
        with open(pdf_path, "wb") as f:
            f.write(contents)
        
        requester = user.get("email") or user.get("uid")
        logger.info(f"PDF saved by {requester}: {pdf_path} ({len(contents)} bytes)")
        
        try:
            msg = agent.ingest_pdf(pdf_path)
            logger.info(f"PDF ingested successfully: {msg}")
            return {"message": msg}
        except Exception as exc:
            error_msg = f"Error ingesting PDF: {str(exc)}"
            logger.error(error_msg)
            return JSONResponse(
                {"error": error_msg}, 
                status_code=500
            )
            
    except Exception as exc:
        error_msg = f"Error uploading PDF: {str(exc)}"
        logger.error(error_msg)
        return JSONResponse(
            {"error": error_msg}, 
            status_code=500
        )


@app.post("/train")
async def train_models(request: Request):
    """Train fault classifier and anomaly detection baseline."""
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
            logger.info(f"Classifier trained: {msg}")
        except Exception as exc:
            error_msg = f"⚠️ Error training classifier: {exc}"
            messages.append(error_msg)
            logger.error(error_msg)
    
    if not normal_path.exists():
        messages.append("⚠️ normal.csv not found. Upload a normal dataset first.")
    else:
        try:
            stats = agent.fit_anomaly_baseline(normal_path)
            msg = f"✅ Fitted anomaly baseline on {normal_path.name} with {len(stats)} features."
            messages.append(msg)
            logger.info(msg)
        except Exception as exc:
            error_msg = f"⚠️ Error fitting anomaly baseline: {exc}"
            messages.append(error_msg)
            logger.error(error_msg)
    
    return {"messages": messages}


# ---------------------------------------------------------------------
# User endpoints
# ---------------------------------------------------------------------

@app.get("/user", response_class=HTMLResponse)
async def user_page():
    """Render the user dashboard with live graph and chat interface."""
    return HTMLResponse(read_template("user.html"))


def _generate_channels(base: float, spread: float, count: int = 4) -> list[float]:
    """Generate random sensor values around a base value."""
    return [round(random.uniform(base - spread, base + spread), 2) for _ in range(count)]


@app.get("/sensor-data")
async def sensor_data(user: dict = Depends(verify_firebase_token)) -> dict:
    """Return simulated sensor data. Requires authentication."""
    data = {
        "voltage": _generate_channels(300.0, 15.0),
        "current": _generate_channels(15.0, 4.0),
        "temperature": _generate_channels(45.0, 20.0),
    }
    logger.debug(f"Sensor data requested by {user.get('email') or user.get('uid')}")
    return data


@app.post("/chat")
async def chat_endpoint(request: Request, user: dict = Depends(verify_firebase_token)):
    """Chat with the local LLM. Requires authentication."""
    try:
        data = await request.json()
        message: Optional[str] = data.get("message")
        sensor_context = data.get("context")
        latest_readings = data.get("latestReadings")
        
        if not message:
            return JSONResponse(
                {"response": "Please provide a message."}, 
                status_code=400
            )

        requester = user.get("email") or user.get("uid")
        logger.info(f"Chat message from {requester}: {message[:50]}...")
        
        # Build enhanced context
        context_parts = []
        if latest_readings:
            context_parts.append("\n[Current Sensor Readings]")
            for sensor_type, values in latest_readings.items():
                context_parts.append(f"{sensor_type.capitalize()}: {values}")
        
        if sensor_context:
            context_parts.append("\n[Recent Data Points]")
            context_parts.append(json.dumps(sensor_context, indent=2))
        
        if context_parts:
            enhanced_message = f"{message}\n{''.join(context_parts)}"
        else:
            enhanced_message = message
        
        response_text = agent.ollama.generate(enhanced_message)
        logger.info(f"LLM response generated ({len(response_text)} chars)")
        return {"response": response_text}
        
    except OllamaNotRunningError as exc:
        error_msg = f"Ollama server is not running. Please start it with 'ollama serve'."
        logger.error(error_msg)
        return JSONResponse({"response": error_msg}, status_code=503)
    except Exception as exc:
        error_msg = f"Chat error: {str(exc)}"
        logger.error(error_msg)
        return JSONResponse({"response": error_msg}, status_code=500)


@app.post("/ask-rag")
async def ask_rag(request: Request, user: dict = Depends(verify_firebase_token)):
    """Query the RAG system. Requires authentication."""
    try:
        data = await request.json()
        question: Optional[str] = data.get("question") or data.get("message")
        
        if not question:
            return JSONResponse(
                {"response": "Please provide a question."}, 
                status_code=400
            )
        
        requester = user.get("email") or user.get("uid")
        logger.info(f"RAG question from {requester}: {question[:50]}...")
        
        answer = agent.query_rag(question)
        logger.info(f"RAG response generated ({len(answer)} chars)")
        return {"response": answer}
        
    except Exception as exc:
        error_msg = f"RAG error: {str(exc)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            {"response": error_msg}, 
            status_code=500
        )


@app.exception_handler(404)
async def not_found(request: Request, exc):
    """Handle 404 errors with a simple message."""
    return JSONResponse({"detail": "Not found"}, status_code=404)


@app.exception_handler(500)
async def internal_error(request: Request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        {"detail": "Internal server error"}, 
        status_code=500
    )


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "ollama": "connected" if agent.ollama else "disconnected",
        "firebase": "configured" if firebase_app else "not configured"
    }