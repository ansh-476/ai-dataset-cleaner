#!/usr/bin/env bash
# AI Dataset Cleaner - start script (macOS / Linux)
# 1) cp .env.example .env and set the provider/key   2) ./run.sh  -> http://localhost:8000
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "[setup] Creating virtual environment..."
  python3 -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install -r requirements.txt
fi
echo "[run] Starting server at http://localhost:8000"
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
