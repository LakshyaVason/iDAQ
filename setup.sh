#!/usr/bin/env bash

# Setup script for iDAQ Diagnostics
#
# This script mirrors the steps documented in README.md to prepare the
# environment, verify prerequisites, pull the Ollama model, and launch the
# FastAPI web application with uvicorn.

set -euo pipefail

MODEL_NAME=${MODEL_NAME:-"llama3.2:1b"}
APP_PORT=${APP_PORT:-8000}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

echo "🔍 Checking prerequisites..."

if ! command_exists python3; then
  echo "❌ Python 3 is required. Please install Python 3.10+ and re-run." >&2
  exit 1
fi

if ! command_exists pip; then
  echo "❌ pip is required. Please install pip (python3 -m ensurepip --upgrade)." >&2
  exit 1
fi

if ! command_exists ollama; then
  cat <<'EOF'
❌ Ollama CLI not found.
Install Ollama from https://ollama.com (or the Jetson-specific build),
then start the service in another terminal with:
  ollama serve
EOF
  exit 1
fi

echo "✅ Python and pip detected"

echo "📦 Creating virtual environment (.venv) if missing..."
python3 -m venv .venv
source .venv/bin/activate

echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🧠 Checking Ollama service and model (${MODEL_NAME})..."
if ! curl -sf http://127.0.0.1:11434/ >/dev/null 2>&1; then
  cat <<'EOF'
⚠️ Ollama does not appear to be running.
Open a new terminal and start it with:
  ollama serve
Then re-run this script.
EOF
  exit 1
fi

if ! ollama list | grep -q "${MODEL_NAME}"; then
  echo "⬇️ Pulling model ${MODEL_NAME} (this may take a while)..."
  ollama pull "${MODEL_NAME}"
else
  echo "✅ Model ${MODEL_NAME} already available"
fi

echo "🚀 Launching FastAPI with uvicorn on port ${APP_PORT}..."
exec uvicorn main:app --reload --host 0.0.0.0 --port "${APP_PORT}"
