#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
pkill -f "streamlit run" || true
PORT=${PORT:-8501}
TARGET="streamlit_app.py"
[ -f "$TARGET" ] || TARGET="pages/00_Login.py"
PYTHONPATH=. STREAMLIT_SERVER_PORT=$PORT streamlit run "$TARGET"
