"""
Complete FastAPI server for iDAQ diagnostics with OpenAI and Firebase integration.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from dotenv import load_dotenv
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials, firestore

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ai_agent import DiagnosticsAgent

from live_data_loader import initialize_data_loader, get_live_data, get_loader_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DATASHEETS_DIR = BASE_DIR / "datasheets"

DATASHEETS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Initialize live data loader
LIVE_DATA_AVAILABLE = initialize_data_loader()

# Firebase setup
firebase_app: Optional[firebase_admin.App] = None
firebase_available: bool = False
db: Optional[firestore.Client] = None

# Initialize AI agent
agent = DiagnosticsAgent()

app = FastAPI(title="iDAQ Diagnostics Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Helper Functions =====

def read_template(name: str) -> str:
    """Read HTML template."""
    path = TEMPLATES_DIR / name
    if not path.exists():
        logger.error(f"Template {name} not found at {path}")
        return f"<html><body><h1>Template {name} not found</h1></body></html>"
    return path.read_text(encoding="utf-8")


def is_admin(request: Request) -> bool:
    """Check admin cookie."""
    return request.cookies.get("admin") == "1"


def redirect_to_login() -> RedirectResponse:
    return RedirectResponse(url="/login", status_code=303)


def init_firebase_admin():
    """Initialize Firebase Admin SDK."""
    global firebase_app, firebase_available, db
    
    if firebase_admin._apps:
        firebase_app = firebase_admin.get_app()
        db = firestore.client()
        firebase_available = True
        return firebase_app
    
    try:
        key_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        key_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_FILE")
        project_id = os.getenv("FIREBASE_PROJECT_ID")
        
        if key_json:
            cred_dict = json.loads(key_json.replace("\\n", "\n"))
            cred = credentials.Certificate(cred_dict)
        elif key_path and Path(key_path).exists():
            cred = credentials.Certificate(key_path)
        else:
            if project_id:
                os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
            cred = credentials.ApplicationDefault()
        
        firebase_app = firebase_admin.initialize_app(cred)
        db = firestore.client()
        firebase_available = True
        return firebase_app
    
    except Exception as e:
        logger.error(f"Firebase initialization failed: {e}")
        firebase_available = False
        return None


def get_firebase_client_config() -> dict:
    """Get Firebase client config."""
    required_keys = ["apiKey", "authDomain", "projectId", "storageBucket", "messagingSenderId", "appId"]
    
    config = {
        "apiKey": os.getenv("FIREBASE_API_KEY"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": os.getenv("FIREBASE_APP_ID"),
    }
    
    measurement_id = os.getenv("FIREBASE_MEASUREMENT_ID")
    if measurement_id:
        config["measurementId"] = measurement_id
    
    missing = [k for k in required_keys if not config.get(k)]
    if missing:
        raise RuntimeError(f"Missing Firebase keys: {', '.join(missing)}")
    
    return config


async def verify_firebase_token(authorization: str = Header(None)) -> dict:
    """Verify Firebase ID token."""
    if not firebase_available:
        return {"uid": "guest", "email": None, "name": "Guest"}
    
    if not authorization or not authorization.startswith("Bearer "):
        return {"uid": "guest", "email": None, "name": "Guest"}
    
    token = authorization.split(" ", 1)[1]
    
    try:
        decoded = firebase_auth.verify_id_token(token)
        return decoded
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        return {"uid": "guest", "email": None, "name": "Guest"}


def save_session_data(user_id: str, session_data: Dict) -> None:
    """Save session data to Firestore."""
    if not db:
        return
    
    try:
        doc_ref = db.collection("sessions").document(user_id).collection("history").document()
        session_data["saved_at"] = firestore.SERVER_TIMESTAMP
        doc_ref.set(session_data)
        logger.info(f"Session saved for user {user_id}")
    except Exception as e:
        logger.error(f"Error saving session: {e}")


def get_user_sessions(user_id: str, limit: int = 10) -> List[Dict]:
    """Retrieve user's session history."""
    if not db:
        return []
    
    try:
        sessions = (
            db.collection("sessions")
            .document(user_id)
            .collection("history")
            .order_by("saved_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        
        return [{"id": s.id, **s.to_dict()} for s in sessions]
    except Exception as e:
        logger.error(f"Error retrieving sessions: {e}")
        return []


# ===== Startup Event =====

@app.on_event("startup")
async def startup_event():
    """Startup checks."""
    logger.info("=" * 60)
    logger.info("iDAQ Diagnostics Server Starting Up")
    logger.info("=" * 60)
    
    # Check OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not found in environment!")
        logger.error("Add it to .env file")
    else:
        logger.info("✅ OpenAI API key configured")
    
    # Check live data
    if LIVE_DATA_AVAILABLE:
        info = get_loader_info()
        logger.info(f"✅ Live data loaded: {info['source_file']} ({info['total_points']} points)")
    else:
        logger.warning("⚠️ No CSV data found - using simulation mode")
        logger.warning("   Place 'VinIinMOSFETVdsSCRVds_240_ALL.csv' in project root for live data")
    
    # Check Firebase
    try:
        init_firebase_admin()
        logger.info("✅ Firebase Admin SDK initialized")
    except Exception as e:
        logger.warning(f"⚠️ Firebase initialization failed: {e}")
        logger.warning("Firebase features will be disabled")
    
    logger.info("=" * 60)


# ===== Authentication Endpoints =====

@app.get("/")
async def index(request: Request):
    """Root redirect."""
    if is_admin(request):
        return RedirectResponse(url="/admin", status_code=303)
    return RedirectResponse(url="/user", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_form():
    """Login form."""
    return HTMLResponse(read_template("login.html"))


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Admin login."""
    if username == "pqlab" and password == "PQ2025!":
        response = RedirectResponse(url="/admin", status_code=303)
        response.set_cookie(key="admin", value="1", httponly=True)
        return response
    return RedirectResponse(url="/user", status_code=303)


@app.get("/logout")
async def logout():
    """Logout."""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key="admin")
    return response


@app.get("/config/firebase")
async def firebase_config():
    """Firebase client config."""
    try:
        return get_firebase_client_config()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== Admin Endpoints =====

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """Admin dashboard."""
    if not is_admin(request):
        return redirect_to_login()
    return HTMLResponse(read_template("admin.html"))


@app.post("/upload-normal")
async def upload_normal(request: Request, file: UploadFile = File(...)):
    """Upload normal dataset."""
    if not is_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    
    try:
        contents = await file.read()
        data_path = BASE_DIR / "normal.csv"
        with open(data_path, "wb") as f:
            f.write(contents)
        logger.info(f"Normal dataset uploaded: {len(contents)} bytes")
        return {"message": "Normal dataset uploaded successfully"}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/upload-fault")
async def upload_fault(request: Request, file: UploadFile = File(...)):
    """Upload fault dataset."""
    if not is_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    
    try:
        contents = await file.read()
        data_path = BASE_DIR / "fault.csv"
        with open(data_path, "wb") as f:
            f.write(contents)
        logger.info(f"Fault dataset uploaded: {len(contents)} bytes")
        return {"message": "Fault dataset uploaded successfully"}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    user: dict = Depends(verify_firebase_token)
):
    """Upload datasheet PDF for RAG."""
    try:
        if not file.filename.endswith('.pdf'):
            return JSONResponse({"error": "Only PDF files allowed"}, status_code=400)
        
        contents = await file.read()
        if len(contents) == 0:
            return JSONResponse({"error": "Empty file"}, status_code=400)
        
        pdf_path = DATASHEETS_DIR / file.filename
        with open(pdf_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"PDF saved: {pdf_path} ({len(contents)} bytes)")
        
        # Ingest into vector store
        msg = agent.ingest_pdf(pdf_path)
        logger.info(f"PDF ingested: {msg}")
        
        return {"message": msg}
    
    except Exception as e:
        logger.error(f"PDF upload error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/train")
async def train_models(request: Request):
    """Train models."""
    if not is_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    
    messages = []
    fault_path = BASE_DIR / "fault.csv"
    normal_path = BASE_DIR / "normal.csv"
    
    if fault_path.exists():
        try:
            msg = agent.train_fault_classifier(fault_path)
            messages.append(msg)
        except Exception as e:
            messages.append(f"⚠️ Classifier training error: {e}")
    else:
        messages.append("⚠️ fault.csv not found")
    
    if normal_path.exists():
        try:
            stats = agent.fit_anomaly_baseline(normal_path)
            messages.append(f"✅ Anomaly baseline fitted with {len(stats)} features")
        except Exception as e:
            messages.append(f"⚠️ Baseline fitting error: {e}")
    else:
        messages.append("⚠️ normal.csv not found")
    
    return {"messages": messages}


# ===== User Endpoints =====

@app.get("/user", response_class=HTMLResponse)
async def user_page():
    """User dashboard."""
    return HTMLResponse(read_template("user.html"))


def _generate_channels(base: float, spread: float, count: int = 4) -> list:
    """Generate random sensor values."""
    return [round(random.uniform(base - spread, base + spread), 2) for _ in range(count)]


@app.get("/sensor-data")
async def sensor_data() -> dict:
    """
    Get sensor data - uses live CSV data if available, otherwise simulation.
    No authentication required for demo mode.
    """
    if LIVE_DATA_AVAILABLE:
        # Use real data from CSV
        data = get_live_data()
    else:
        # Fallback to random simulation
        data = {
            "voltage": _generate_channels(300.0, 15.0),
            "current": _generate_channels(15.0, 4.0),
            "temperature": _generate_channels(45.0, 20.0),
        }
    
    return data


@app.post("/chat")
async def chat_endpoint(request: Request, user: dict = Depends(verify_firebase_token)):
    """Chat with OpenAI."""
    try:
        data = await request.json()
        message = data.get("message")
        session_data = data.get("context")
        latest_readings = data.get("latestReadings")
        session_info = data.get("sessionInfo", {})
        
        if not message:
            return JSONResponse({"response": "Please provide a message"}, status_code=400)
        
        logger.info(f"Chat from {user.get('email') or user.get('uid')}: {message[:50]}...")
        
        # Enhanced context for time-based queries
        context_note = ""
        if session_data:
            context_note = f"""
You have access to sensor data with timestamps. When asked about specific times:
- Look for the 'time' field in the format HH:MM:SS
- Match the time in the user's question to the 'time' field in the data
- Each data point has: time, timestamp, voltage (array), current (array), temperature (array)

Example: If asked "What was the voltage at 14:30:15?", find the entry with time="14:30:15" and report its voltage values.

Current session has {len(session_data)} data points spanning from {session_data[0]['time'] if session_data else 'N/A'} to {session_data[-1]['time'] if session_data else 'N/A'}.
"""
        
        response_text = agent.chat(
            message=message + context_note,
            session_data=session_data,
            latest_readings=latest_readings
        )
        
        logger.info(f"Response generated ({len(response_text)} chars)")
        return {"response": response_text}
    
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return JSONResponse({"response": f"Error: {str(e)}"}, status_code=500)


@app.post("/ask-rag")
async def ask_rag(request: Request, user: dict = Depends(verify_firebase_token)):
    """Query RAG system."""
    try:
        data = await request.json()
        question = data.get("question") or data.get("message")
        session_data = data.get("context")
        
        if not question:
            return JSONResponse({"response": "Please provide a question"}, status_code=400)
        
        logger.info(f"RAG query from {user.get('email') or user.get('uid')}: {question[:50]}...")
        logger.info(f"Vector store status: {agent.vector_store is not None}")
        
        # Check if vector store exists
        if agent.vector_store is None:
            logger.warning("Vector store is None - no PDFs uploaded yet")
            return {"response": "⚠️ No datasheets uploaded yet. Please:\n1. Make sure you're in RAG mode\n2. Upload a PDF using the file selector\n3. Wait for the 'Ingested X chunks' message\n4. Then try your question again"}
        
        answer = agent.query_rag(question, session_data)
        
        logger.info(f"RAG response generated ({len(answer)} chars)")
        return {"response": answer}
    
    except Exception as e:
        logger.error(f"RAG error: {e}", exc_info=True)
        return JSONResponse({"response": f"❌ RAG Error: {str(e)}\n\nMake sure you:\n1. Uploaded a PDF in RAG mode\n2. Waited for ingestion confirmation\n3. Are asking about content in the PDF"}, status_code=500)


@app.post("/save-session")
async def save_session(request: Request, user: dict = Depends(verify_firebase_token)):
    """Save user session to Firebase."""
    try:
        data = await request.json()
        session_info = {
            "user_id": user.get("uid"),
            "user_email": user.get("email"),
            "session_data": data.get("sessionData", []),
            "chat_history": data.get("chatHistory", []),
            "start_time": data.get("startTime"),
            "end_time": data.get("endTime"),
            "mode": data.get("mode", "live")
        }
        
        save_session_data(user.get("uid"), session_info)
        
        return {"message": "Session saved successfully"}
    
    except Exception as e:
        logger.error(f"Session save error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/sessions", response_class=HTMLResponse)
async def sessions_page():
    """Sessions history page."""
    return HTMLResponse(read_template("sessions.html"))


@app.get("/user-sessions")
async def user_sessions(user: dict = Depends(verify_firebase_token)):
    """Get user's session history."""
    try:
        sessions = get_user_sessions(user.get("uid"))
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Session retrieval error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/delete-session/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(verify_firebase_token)):
    """Delete a user session."""
    if not db:
        return JSONResponse({"error": "Firestore not available"}, status_code=503)
    
    try:
        doc_ref = db.collection("sessions").document(user.get("uid")).collection("history").document(session_id)
        doc_ref.delete()
        logger.info(f"Session {session_id} deleted for user {user.get('uid')}")
        return {"message": "Session deleted successfully"}
    except Exception as e:
        logger.error(f"Session deletion error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health_check():
    """Health check."""
    loader_info = get_loader_info() if LIVE_DATA_AVAILABLE else {"loaded": False}
    
    return {
        "status": "ok",
        "openai": "configured" if os.getenv("OPENAI_API_KEY") else "missing",
        "firebase": "configured" if firebase_app else "not configured",
        "vector_store": "loaded" if agent.vector_store else "empty",
        "live_data": loader_info
    }


@app.exception_handler(404)
async def not_found(request: Request, exc):
    """404 handler."""
    return JSONResponse({"detail": "Not found"}, status_code=404)


@app.exception_handler(500)
async def internal_error(request: Request, exc):
    """500 handler."""
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse({"detail": "Internal server error"}, status_code=500)