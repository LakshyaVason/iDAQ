#!/usr/bin/env bash

# Minimal setup script for iDAQ Diagnostics (Git Bash on Windows)
# No venv - just pull model and run

set -euo pipefail

MODEL_NAME=${MODEL_NAME:-"llama3.2:1b"}
APP_PORT=${APP_PORT:-8000}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

echo "Checking prerequisites..."

if ! command_exists python; then
  echo "Python is required. Please install Python 3.10+"
  exit 1
fi

if ! command_exists pip; then
  echo "pip is required to install Python dependencies"
  exit 1
fi

if ! command_exists ollama; then
  echo "Ollama CLI not found."
  echo "Install Ollama from https://ollama.com"
  exit 1
fi

echo "Python, pip, and Ollama detected"

echo "Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Checking Ollama service..."
if ! curl -sf http://127.0.0.1:11434/ >/dev/null 2>&1; then
  echo "Ollama does not appear to be running."
  echo "Open another terminal and run: ollama serve"
  echo "Then re-run this script."
  exit 1
fi

echo "Ollama service is running"

if ! ollama list | grep -q "${MODEL_NAME}"; then
  echo "Pulling model ${MODEL_NAME} (this may take a while)..."
  ollama pull "${MODEL_NAME}"
else
  echo "Model ${MODEL_NAME} already available"
fi

echo ""
echo "Launching FastAPI on port ${APP_PORT}..."
echo "Access at: http://localhost:${APP_PORT}"
echo "Press Ctrl+C to stop"
echo ""
uvicorn main:app --reload --host 0.0.0.0 --port "${APP_PORT}"