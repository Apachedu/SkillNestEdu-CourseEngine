#!/bin/bash
# Generated under WORKING_PROTOCOL.md — Option C — smart runner
set -euo pipefail

# Go to repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "▶️  SkillNestEdu runner starting…"

# 1) Ensure venv
if [ ! -d ".venv" ]; then
  echo "�� Creating .venv…"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "✅ .venv active"

# 2) Install deps if streamlit missing
if ! python -c "import streamlit" >/dev/null 2>&1; then
  echo "📚 Installing Python dependencies…"
  pip install --upgrade pip >/dev/null
  pip install -r requirements.txt
else
  echo "✅ Dependencies OK"
fi

# 3) Logo notice (non-blocking)
if [ ! -f "assets/skillnestlogo.png" ]; then
  echo "ℹ️  Logo not found at assets/skillnestlogo.png — sidebar will show text branding."
fi

# 4) Diagnostics (Option C) — auto-detect Ollama unless overridden
: "${OLLAMA:=auto}"             # auto|0|1
: "${OLLAMA_MODEL:=llama3}"
if [ "$OLLAMA" = "auto" ]; then
  if command -v ollama >/dev/null 2>&1; then OLLAMA=1; else OLLAMA=0; fi
fi
export OLLAMA OLLAMA_MODEL
if [ "$OLLAMA" = "1" ]; then
  echo "🧠 Diagnostics: rules + Ollama (${OLLAMA_MODEL})"
else
  echo "🛡️  Diagnostics: rules-only (Ollama disabled/not found)"
fi

# 5) Launch Streamlit
PORT="${PORT:-8501}"
echo "🌐 Launching http://localhost:${PORT}"
exec streamlit run streamlit_app.py --server.port="${PORT}"
