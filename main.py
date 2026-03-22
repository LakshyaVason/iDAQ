"""
Complete FastAPI server for iDAQ diagnostics with OpenAI and Firebase integration.
Now with C2000 UART integration.
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

# Try to import C2000 serial reader first, fallback to CSV loader
try:
    from c2000_serial_reader import (
        initialize_c2000_reader, 
        get_c2000_data, 
        is_c2000_connected,
        get_c2000_stats,
        get_c2000_reader_recent 
    )
    C2000_AVAILABLE = True
except ImportError:
    C2000_AVAILABLE = False
    print("⚠️ c2000_serial_reader not found - will use CSV fallback")

# CSV data loader as fallback

from live_data_loader import initialize_data_loader, get_live_data, get_loader_info, get_live_batch
try:
    from live_data_loader import initialize_data_loader, get_live_data, get_loader_info
    CSV_AVAILABLE = initialize_data_loader()
except ImportError:
    CSV_AVAILABLE = False
    print("⚠️ live_data_loader not found - will use simulation")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DATASHEETS_DIR = BASE_DIR / "datasheets"

DATASHEETS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Firebase setup
firebase_app: Optional[firebase_admin.App] = None
firebase_available: bool = False
db: Optional[firestore.Client] = None

# Initialize AI agent
agent = DiagnosticsAgent()

# Global pause state for data streaming
data_streaming_paused = False

# Data source tracking
current_data_source = "unknown"  # Will be set to "c2000", "csv", or "simulation"

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
    """Startup checks and initialization."""
    global current_data_source
    
    logger.info("=" * 60)
    logger.info("iDAQ Diagnostics Server Starting Up")
    logger.info("=" * 60)
    
    # Check OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not found in environment!")
        logger.error("Add it to .env file")
    else:
        logger.info("✅ OpenAI API key configured")
    
    # Initialize data source (priority: C2000 → CSV → Simulation)
    if C2000_AVAILABLE:
        # Try to connect to C2000 via UART
        uart_port = os.getenv("C2000_UART_PORT", "/dev/ttyTHS1")
        uart_baud = int(os.getenv("C2000_UART_BAUD", "115200"))
        
        logger.info(f"🔌 Attempting C2000 connection on {uart_port} @ {uart_baud} baud...")
        
        if initialize_c2000_reader(port=uart_port, baudrate=uart_baud):
            current_data_source = "c2000"
            logger.info("✅ C2000 UART connected - using live hardware data")
        else:
            logger.warning("⚠️ C2000 connection failed - falling back to CSV/simulation")
            current_data_source = "csv" if CSV_AVAILABLE else "simulation"
    else:
        current_data_source = "csv" if CSV_AVAILABLE else "simulation"
    
    # Log CSV status
    if CSV_AVAILABLE and current_data_source != "c2000":
        try:
            info = get_loader_info()
            logger.info(f"✅ CSV data loaded: {info['source_file']} ({info['total_points']} points)")
        except:
            logger.warning("⚠️ CSV loader available but no data loaded")
    elif current_data_source == "simulation":
        logger.warning("⚠️ Using simulation mode - random data generation")
    
    # Check Firebase
    try:
        init_firebase_admin()
        logger.info("✅ Firebase Admin SDK initialized")
    except Exception as e:
        logger.warning(f"⚠️ Firebase initialization failed: {e}")
        logger.warning("Firebase features will be disabled")
    
    logger.info(f"📊 Active data source: {current_data_source.upper()}")
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
    """Generate random sensor values for simulation mode."""
    return [round(random.uniform(base - spread, base + spread), 2) for _ in range(count)]


@app.get("/sensor-data")
async def sensor_data() -> dict:
    """
    Get sensor data with automatic source selection:
    1. C2000 UART (if connected)
    2. CSV playback (if available)
    3. Simulation (fallback)
    
    Returns 12 channels: 4 voltages, 4 currents, 4 temperatures
    """
    global data_streaming_paused, current_data_source
    
    if data_streaming_paused:
        return {"paused": True}
    
    # Priority 1: C2000 hardware
    if current_data_source == "c2000" and C2000_AVAILABLE:
        try:
            if is_c2000_connected():
                data = get_c2000_data()
                data["paused"] = False
                data["source"] = "c2000"
                return data
            else:
                logger.warning("C2000 disconnected, switching to fallback")
                current_data_source = "csv" if CSV_AVAILABLE else "simulation"
        except Exception as e:
            logger.error(f"C2000 read error: {e}")
            current_data_source = "csv" if CSV_AVAILABLE else "simulation"
    
    # Priority 2: CSV playback
    if current_data_source == "csv" and CSV_AVAILABLE:
        try:
            data = get_live_data()
            data["paused"] = False
            data["source"] = "csv"
            return data
        except Exception as e:
            logger.error(f"CSV read error: {e}")
            current_data_source = "simulation"
    
    # Priority 3: Simulation
    data = {
        "voltage": _generate_channels(300.0, 15.0),
        "current": _generate_channels(15.0, 4.0),
        "temperature": _generate_channels(45.0, 20.0),
        "paused": False,
        "source": "simulation"
    }
    return data

@app.get("/sensor-data-batch")
async def sensor_data_batch(n: int = 50) -> dict:
    """
    Get a batch of n sensor readings for waveform-accurate display.
    Returns array of samples so frontend can plot full AC waveform shape.
    n is capped at 200 to prevent oversized responses.
    """
    global data_streaming_paused, current_data_source

    if data_streaming_paused:
        return {"paused": True, "samples": []}

    n = min(n, 200)  # safety cap

    # C2000 live: return n individual readings from buffer
    if current_data_source == "c2000" and C2000_AVAILABLE:
        try:
            if is_c2000_connected():
                recent = get_c2000_reader_recent(n)  # see note below
                if recent:
                    return {"paused": False, "source": "c2000", "samples": recent}
        except Exception as e:
            logger.error(f"C2000 batch read error: {e}")

    # CSV playback: return batch
    if current_data_source == "csv" and CSV_AVAILABLE:
        try:
            samples = get_live_batch(n)
            return {"paused": False, "source": "csv", "samples": samples}
        except Exception as e:
            logger.error(f"CSV batch read error: {e}")

    # Simulation fallback
    samples = get_live_batch(n)
    return {"paused": False, "source": "simulation", "samples": samples}

@app.post("/pause-streaming")
async def pause_streaming():
    """Pause data streaming."""
    global data_streaming_paused
    data_streaming_paused = True
    logger.info("Data streaming paused")
    return {"status": "paused", "message": "Data streaming paused"}


@app.post("/resume-streaming")
async def resume_streaming():
    """Resume data streaming."""
    global data_streaming_paused
    data_streaming_paused = False
    logger.info("Data streaming resumed")
    return {"status": "resumed", "message": "Data streaming resumed"}


@app.get("/streaming-status")
async def streaming_status():
    """Get current streaming status."""
    global data_streaming_paused
    return {"paused": data_streaming_paused}


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
            "mode": data.get("mode", "live"),
            "data_source": current_data_source
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
    """Health check with data source info."""
    global data_streaming_paused, current_data_source
    
    health_info = {
        "status": "ok",
        "openai": "configured" if os.getenv("OPENAI_API_KEY") else "missing",
        "firebase": "configured" if firebase_app else "not configured",
        "vector_store": "loaded" if agent.vector_store else "empty",
        "data_source": current_data_source,
        "streaming_paused": data_streaming_paused
    }
    
    # Add C2000 stats if available
    if C2000_AVAILABLE and current_data_source == "c2000":
        try:
            health_info["c2000_stats"] = get_c2000_stats()
        except:
            pass
    
    # Add CSV info if available
    if CSV_AVAILABLE and current_data_source == "csv":
        try:
            health_info["csv_info"] = get_loader_info()
        except:
            pass
    
    return health_info


@app.get("/data-source")
async def get_data_source():
    """Get current data source information."""
    global current_data_source
    
    info = {
        "current_source": current_data_source,
        "available_sources": []
    }
    
    if C2000_AVAILABLE:
        info["available_sources"].append({
            "name": "c2000",
            "connected": is_c2000_connected() if current_data_source == "c2000" else False,
            "description": "Live C2000 DSP via UART"
        })
    
    if CSV_AVAILABLE:
        info["available_sources"].append({
            "name": "csv",
            "description": "CSV file playback"
        })
    
    info["available_sources"].append({
        "name": "simulation",
        "description": "Random data generation"
    })
    
    return info


@app.exception_handler(404)
async def not_found(request: Request, exc):
    """404 handler."""
    return JSONResponse({"detail": "Not found"}, status_code=404)


@app.exception_handler(500)
async def internal_error(request: Request, exc):
    """500 handler."""
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse({"detail": "Internal server error"}, status_code=500)