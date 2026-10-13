@echo off
REM ------------------------------------------------------------
REM  AI Dataset Cleaner - start script (Windows)
REM  1) Copy .env.example to .env and set LLM_PROVIDER / LLM_API_KEY (or use ollama)
REM  2) For ollama: ollama pull llama3.2:3b
REM  3) Double-click this file, then open http://localhost:8000
REM ------------------------------------------------------------
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    python -m venv .venv
    echo [setup] Installing dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

echo [run] Starting server at http://localhost:8000
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
