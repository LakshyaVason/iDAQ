"""
Complete FastAPI server for iDAQ diagnostics with OpenAI and Firebase integration.

Updates from original:
- WebSocket endpoint /ws/sensor for <1s live streaming (primary transport)
- Firebase RTDB pusher started at startup (0.1s / 10 Hz, safe for demo on free tier)
- POST /auto-train  → trains RandomForest from all F0-F7 CSVs in training_data/
- POST /upload-training-batch → saves multiple labeled CSVs to training_data/
- GET  /sensor-data → now also includes fault classification dict
- Baud rate default updated to 921600 in c2000_serial_reader.py (not here)
"""

import asyncio
import os
import json
import random
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from dotenv import load_dotenv
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials, firestore

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ai_agent import DiagnosticsAgent
from live_data_loader import initialize_data_loader, get_live_data, get_loader_info
from dataset_manager import run_auto_training

# Firebase pusher (only imported/started if FIREBASE_RTDB_URL is configured)
USE_FIREBASE_PUSHER = bool(os.getenv("FIREBASE_RTDB_URL", ""))
if USE_FIREBASE_PUSHER:
    from firebase_pusher import FirebasePusher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DATASHEETS_DIR = BASE_DIR / "datasheets"
TRAINING_DIR = BASE_DIR / "training_data"

DATASHEETS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

# Initialize live data loader
LIVE_DATA_AVAILABLE = initialize_data_loader()

# Firebase setup
firebase_app: Optional[firebase_admin.App] = None
firebase_available: bool = False
db: Optional[firestore.Client] = None

# Firebase RTDB pusher instance
firebase_pusher: Optional["FirebasePusher"] = None

# Initialize AI agent
agent = DiagnosticsAgent()

# Global pause state for data streaming
data_streaming_paused = False


# ===== WebSocket Manager =====

class WSManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = WSManager()


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


# ===== App =====

app = FastAPI(title="iDAQ Diagnostics Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Background Tasks =====

async def _ws_broadcast_loop():
    """Push sensor data to all connected WebSocket clients at 10 Hz."""
    interval = float(os.getenv("WS_PUSH_INTERVAL", "0.1"))
    while True:
        await asyncio.sleep(interval)
        if not ws_manager.active or data_streaming_paused:
            continue
        try:
            data = get_live_data()
            # Add fault classification if classifier is loaded
            if agent.classifier is not None:
                try:
                    sensor_row = {
                        "Vin": data["voltage"][0],
                        "Iin": data["current"][0],
                        "MOSFET_Vds": data["voltage"][1],
                        "SCR_Vds": data["voltage"][2],
                    }
                    data["fault"] = agent.classify_fault(sensor_row)
                except Exception:
                    data["fault"] = None
            await ws_manager.broadcast(data)
        except Exception as e:
            logger.warning(f"WS broadcast error: {e}")


# ===== Startup Event =====

@app.on_event("startup")
async def startup_event():
    """Startup checks."""
    global firebase_pusher

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

    # Check Firebase Admin
    try:
        init_firebase_admin()
        logger.info("✅ Firebase Admin SDK initialized")
    except Exception as e:
        logger.warning(f"⚠️ Firebase initialization failed: {e}")
        logger.warning("Firebase features will be disabled")

    # Try loading existing classifier
    try:
        agent.load_classifier()
        logger.info("✅ Fault classifier loaded from artifacts/")
    except FileNotFoundError:
        logger.warning("⚠️ No classifier found — upload CSVs and run /auto-train")

    # Firebase RTDB pusher (only if configured)
    if USE_FIREBASE_PUSHER:
        try:
            firebase_pusher = FirebasePusher(data_fn=get_live_data)
            firebase_pusher.start()
            logger.info(f"✅ Firebase RTDB pusher started at {1/float(os.getenv('RTDB_PUSH_INTERVAL','0.1')):.0f} Hz")
        except Exception as e:
            logger.warning(f"⚠️ Firebase pusher failed to start: {e}")

    # Start WebSocket broadcaster
    asyncio.create_task(_ws_broadcast_loop())

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
        msg = agent.ingest_pdf(pdf_path)
        logger.info(f"PDF ingested: {msg}")

        return {"message": msg}

    except Exception as e:
        logger.error(f"PDF upload error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/train")
async def train_models(request: Request):
    """Train models (original single-file method)."""
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


# ===== NEW: Batch Training Endpoints =====

@app.post("/upload-training-batch")
async def upload_training_batch(request: Request, files: List[UploadFile] = File(...)):
    """Upload multiple labeled CSV files (F0_*.csv … F7_*.csv) to training_data/."""
    if not is_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    saved, errors = [], []
    for file in files:
        if not file.filename.endswith(".csv"):
            errors.append(f"{file.filename}: not a CSV")
            continue
        dest = TRAINING_DIR / file.filename
        try:
            contents = await file.read()
            with open(dest, "wb") as f:
                f.write(contents)
            saved.append(file.filename)
        except Exception as e:
            errors.append(f"{file.filename}: {e}")

    return {
        "saved": saved,
        "errors": errors,
        "total": len(saved),
        "message": f"Saved {len(saved)} files. Run /auto-train to retrain.",
    }


@app.post("/auto-train")
async def auto_train(request: Request):
    """Scan training_data/, train RandomForest on all F0-F7 classes, hot-reload into agent."""
    if not is_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    try:
        result = run_auto_training(TRAINING_DIR)
        # Hot-reload the new classifier into the live agent
        try:
            agent.load_classifier()
            result["classifier_reloaded"] = True
        except Exception as e:
            result["classifier_reload_error"] = str(e)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    Returns cached/paused data when streaming is paused.
    Now also includes fault classification if a model is trained.
    """
    global data_streaming_paused

    if data_streaming_paused:
        return {"paused": True}

    if LIVE_DATA_AVAILABLE:
        data = get_live_data()
    else:
        data = {
            "voltage": _generate_channels(300.0, 15.0),
            "current": _generate_channels(15.0, 4.0),
            "temperature": _generate_channels(45.0, 20.0),
        }

    data["paused"] = False

    # Add fault classification if classifier is loaded
    if agent.classifier is not None:
        try:
            sensor_row = {
                "Vin": data["voltage"][0],
                "Iin": data["current"][0],
                "MOSFET_Vds": data["voltage"][1],
                "SCR_Vds": data["voltage"][2],
            }
            data["fault"] = agent.classify_fault(sensor_row)
        except Exception as e:
            logger.warning(f"Classification error: {e}")
            data["fault"] = None
    else:
        data["fault"] = None

    return data


# ===== NEW: WebSocket Endpoint =====

@app.websocket("/ws/sensor")
async def ws_sensor(websocket: WebSocket):
    """
    WebSocket for live sensor data. Primary transport (~10 Hz, <50ms on LAN).
    The frontend falls back to Firebase RTDB onValue() if this is unreachable
    (e.g. phone on a different network).
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if msg == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# ===== Streaming Control =====

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


# ===== Chat / RAG Endpoints =====

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

        if agent.vector_store is None:
            logger.warning("Vector store is None - no PDFs uploaded yet")
            return {"response": "⚠️ No datasheets uploaded yet. Please:\n1. Make sure you're in RAG mode\n2. Upload a PDF using the file selector\n3. Wait for the 'Ingested X chunks' message\n4. Then try your question again"}

        answer = agent.query_rag(question, session_data)

        logger.info(f"RAG response generated ({len(answer)} chars)")
        return {"response": answer}

    except Exception as e:
        logger.error(f"RAG error: {e}", exc_info=True)
        return JSONResponse({"response": f"❌ RAG Error: {str(e)}\n\nMake sure you:\n1. Uploaded a PDF in RAG mode\n2. Waited for ingestion confirmation\n3. Are asking about content in the PDF"}, status_code=500)


# ===== Session Endpoints =====

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


# ===== Health =====

@app.get("/health")
async def health_check():
    """Health check."""
    global data_streaming_paused
    loader_info = get_loader_info() if LIVE_DATA_AVAILABLE else {"loaded": False}

    return {
        "status": "ok",
        "openai": "configured" if os.getenv("OPENAI_API_KEY") else "missing",
        "firebase": "configured" if firebase_app else "not configured",
        "firebase_pusher": firebase_pusher.stats if firebase_pusher else "disabled",
        "vector_store": "loaded" if agent.vector_store else "empty",
        "classifier": "loaded" if agent.classifier else "not trained",
        "live_data": loader_info,
        "streaming_paused": data_streaming_paused,
        "ws_connections": len(ws_manager.active),
    }


# ===== Error Handlers =====

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse({"detail": "Not found"}, status_code=404)


@app.exception_handler(500)
async def internal_error(request: Request, exc):
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse({"detail": "Internal server error"}, status_code=500)