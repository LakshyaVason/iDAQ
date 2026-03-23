"""
firebase_pusher.py

Pushes live iDAQ sensor data to Firebase Realtime Database so it can be
viewed from any device (phone, laptop) at https://idaq-diagnostics.web.app

Reads credentials from your existing .env and serviceAccountKey.json.
No new config needed beyond FIREBASE_DATABASE_URL in .env.

Usage:
    Called automatically from main.py startup_event().
    Standalone test:
        python3 firebase_pusher.py
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request as GARequest
from google.oauth2 import service_account

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config from .env ───────────────────────────────────────────────────────────
RTDB_URL             = os.getenv("FIREBASE_DATABASE_URL", "").rstrip("/")
SERVICE_ACCOUNT_FILE = os.getenv("FIREBASE_SERVICE_ACCOUNT_FILE", "serviceAccountKey.json")
PUSH_INTERVAL        = float(os.getenv("FIREBASE_PUSH_INTERVAL", "0.5"))  # 2 Hz default

SCOPES = [
    "https://www.googleapis.com/auth/firebase.database",
    "https://www.googleapis.com/auth/userinfo.email",
]


class FirebasePusher:
    """
    Background thread that pushes sensor data to Firebase RTDB.

    Writes to:
      /live    — latest reading (real-time chart on phone)
      /status  — uptime, write count, data source
    """

    def __init__(self, data_fn: Callable[[], Dict]):
        if not RTDB_URL:
            raise ValueError(
                "FIREBASE_DATABASE_URL not set in .env\n"
                "Add: FIREBASE_DATABASE_URL=https://idaq-diagnostics-default-rtdb.firebaseio.com"
            )

        self.data_fn       = data_fn
        self._creds        = None
        self._token        = None
        self._token_expiry = 0.0
        self._running      = False
        self._thread       = None
        self._writes       = 0
        self._errors       = 0
        self._start_time   = None

    def _get_token(self) -> str:
        now = time.time()
        if self._token and now < self._token_expiry - 60:
            return self._token
        if self._creds is None:
            self._creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )
        self._creds.refresh(GARequest())
        self._token = self._creds.token
        expiry = self._creds.expiry
        self._token_expiry = expiry.timestamp() if expiry else (time.time() + 3600)
        return self._token

    def _push(self, data: Dict) -> None:
        token   = self._get_token()
        payload = {
            **data,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "sample_count": self._writes,
            "uptime_s":     round(time.time() - self._start_time, 1) if self._start_time else 0,
        }
        r = requests.put(f"{RTDB_URL}/live.json?auth={token}", json=payload, timeout=5)
        r.raise_for_status()
        self._writes += 1

    def _push_status(self, source: str) -> None:
        try:
            token = self._get_token()
            requests.put(
                f"{RTDB_URL}/status.json?auth={token}",
                json={
                    "online":       True,
                    "source":       source,
                    "last_seen":    datetime.now(timezone.utc).isoformat(),
                    "total_writes": self._writes,
                    "errors":       self._errors,
                },
                timeout=5,
            )
        except Exception:
            pass

    def _loop(self) -> None:
        self._start_time = time.time()
        status_tick = 0

        while self._running:
            t0 = time.perf_counter()
            try:
                data = self.data_fn()
                self._push(data)
                status_tick += 1
                if status_tick >= 10:
                    self._push_status(data.get("source", "unknown"))
                    status_tick = 0
            except Exception as e:
                self._errors += 1
                if self._errors % 20 == 1:
                    logger.error(f"[FirebasePusher] error #{self._errors}: {e}")

            time.sleep(max(0.0, PUSH_INTERVAL - (time.perf_counter() - t0)))

        # Mark offline when stopped
        try:
            token = self._get_token()
            requests.put(
                f"{RTDB_URL}/status.json?auth={token}",
                json={"online": False, "last_seen": datetime.now(timezone.utc).isoformat()},
                timeout=5,
            )
        except Exception:
            pass

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="FirebasePusher")
        self._thread.start()
        print(f"✅ Firebase pusher started at {1/PUSH_INTERVAL:.1f} Hz → {RTDB_URL}/live")

    def stop(self) -> None:
        self._running = False

    @property
    def stats(self) -> Dict:
        return {
            "writes":   self._writes,
            "errors":   self._errors,
            "rate_hz":  round(1.0 / PUSH_INTERVAL, 2),
            "rtdb_url": RTDB_URL,
        }


# ── Global helpers (called from main.py) ──────────────────────────────────────

_pusher: Optional[FirebasePusher] = None


def start_firebase_pusher(data_fn: Callable[[], Dict]) -> bool:
    global _pusher
    try:
        _pusher = FirebasePusher(data_fn=data_fn)
        _pusher.start()
        return True
    except Exception as e:
        logger.error(f"[FirebasePusher] Failed to start: {e}")
        print(f"⚠️  Firebase pusher failed: {e}")
        return False


def stop_firebase_pusher() -> None:
    if _pusher:
        _pusher.stop()


def get_pusher_stats() -> Dict:
    return _pusher.stats if _pusher else {"running": False}


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math, random

    print("Firebase Pusher — standalone test")
    print("=" * 50)
    print(f"RTDB URL : {RTDB_URL or '❌ NOT SET in .env'}")
    print(f"Key file : {SERVICE_ACCOUNT_FILE}")
    print()

    if not RTDB_URL:
        print("Set FIREBASE_DATABASE_URL in .env first")
        exit(1)

    def fake_sensor() -> Dict:
        t = time.time()
        return {
            "voltage":     [round(120 * abs(math.sin(t * 2 * math.pi * 0.5)), 2), 0, 0, 0],
            "current":     [round(10 + math.sin(t) * 2, 2), 0, 0, 0],
            "temperature": [round(35 + random.uniform(-1, 1), 1), 25, 25, 0],
            "source":      "test",
        }

    pusher = FirebasePusher(data_fn=fake_sensor)
    pusher.start()

    print(f"Pushing for 10s — view at: {RTDB_URL}/live.json")
    for i in range(10):
        time.sleep(1)
        s = pusher.stats
        print(f"  [{i+1:2d}s] writes={s['writes']:3d}  errors={s['errors']}")

    pusher.stop()
    print("Done")