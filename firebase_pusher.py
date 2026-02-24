"""
Firebase Realtime DB pusher for iDAQ live sensor data.

Runs as a background thread on the Jetson, pushing sensor readings
to Firebase RTDB at RTDB_PUSH_INTERVAL seconds (default 0.1s = 10 Hz).

For a short demo session at 10 Hz:
  2 hours × 3600s × 10 Hz = 72,000 writes — well within Firebase free tier (180k/day).

Prerequisites:
  pip install google-auth google-auth-requests requests
  FIREBASE_SERVICE_ACCOUNT_FILE=serviceAccountKey.json in .env
  FIREBASE_RTDB_URL=https://YOUR-PROJECT-default-rtdb.firebaseio.com in .env
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict

import requests
from google.auth.transport.requests import Request as GARequest
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

RTDB_URL             = os.getenv(
    "FIREBASE_RTDB_URL",
    "https://idaq-diagnostics-default-rtdb.firebaseio.com"
)
PUSH_INTERVAL_SEC    = float(os.getenv("RTDB_PUSH_INTERVAL", "0.1"))  # 10 Hz default
SERVICE_ACCOUNT_FILE = os.getenv("FIREBASE_SERVICE_ACCOUNT_FILE", "serviceAccountKey.json")
SCOPES = [
    "https://www.googleapis.com/auth/firebase.database",
    "https://www.googleapis.com/auth/userinfo.email",
]


class FirebasePusher:
    """
    Pushes sensor data to Firebase Realtime Database at a fixed rate.

    Usage:
        pusher = FirebasePusher(data_fn=get_live_data)
        pusher.start()
        # ... runs in background ...
        pusher.stop()
    """

    def __init__(self, data_fn: Callable[[], Dict]):
        """
        Parameters
        ----------
        data_fn : callable that returns the current sensor dict
                  {"voltage": [...], "current": [...], "temperature": [...]}
        """
        self.data_fn       = data_fn
        self._creds        = None
        self._token        = None
        self._token_expiry = 0.0
        self._running      = False
        self._thread       = None
        self._writes       = 0
        self._errors       = 0

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    def _get_token(self) -> str:
        """Return a valid OAuth2 access token, refreshing if needed."""
        now = time.time()
        if self._token and now < self._token_expiry - 60:
            return self._token

        if self._creds is None:
            self._creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )
        self._creds.refresh(GARequest())
        self._token        = self._creds.token
        self._token_expiry = self._creds.expiry.timestamp()
        return self._token

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------
    def _push(self, data: Dict) -> None:
        """PATCH /live node with latest sensor reading (single DB write)."""
        token   = self._get_token()
        ts      = datetime.now(timezone.utc).isoformat()
        payload = {**data, "timestamp": ts, "seq": self._writes}
        url     = f"{RTDB_URL}/live.json?auth={token}"
        r       = requests.patch(url, json=payload, timeout=3)
        r.raise_for_status()
        self._writes += 1

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            t0 = time.perf_counter()
            try:
                data = self.data_fn()
                self._push(data)
            except Exception as e:
                self._errors += 1
                # Log every 20th error to avoid spam
                if self._errors % 20 == 1:
                    logger.error(f"[FirebasePusher] push error ({self._errors} total): {e}")

            # Precise sleep: subtract time already spent pushing
            elapsed = time.perf_counter() - t0
            sleep_s = max(0.0, PUSH_INTERVAL_SEC - elapsed)
            time.sleep(sleep_s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background push thread."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="FirebasePusher"
        )
        self._thread.start()
        hz = 1.0 / PUSH_INTERVAL_SEC
        logger.info(f"[FirebasePusher] Started at {hz:.1f} Hz → {RTDB_URL}/live")
        print(f"✅ FirebasePusher started at {hz:.1f} Hz")

    def stop(self) -> None:
        """Stop the background push thread."""
        self._running = False

    @property
    def stats(self) -> Dict:
        """Return write/error counts."""
        return {
            "writes":   self._writes,
            "errors":   self._errors,
            "rate_hz":  round(1.0 / PUSH_INTERVAL_SEC, 2),
        }