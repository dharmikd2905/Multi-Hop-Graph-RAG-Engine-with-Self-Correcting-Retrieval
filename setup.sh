#!/usr/bin/env bash
# Zero-cost local reproduction setup (PRD section 2, pillar 1).
#
# Usage:
#   ./setup.sh            # sets up venv + deps + local Ollama models
#   ./setup.sh --no-ollama  # skip Ollama model pulls (use openai/groq instead)

set -euo pipefail

echo "==> Graph-Augmented Self-RAG Engine -- local setup"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Creating virtual environment (.venv)"
"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example"
  cp .env.example .env
else
  echo "==> .env already exists, leaving it untouched"
fi

mkdir -p data/chroma

if [[ "${1:-}" != "--no-ollama" ]]; then
  if command -v ollama >/dev/null 2>&1; then
    echo "==> Ollama detected. Pulling models (llama3.1, nomic-embed-text)..."
    ollama pull llama3.1
    ollama pull nomic-embed-text
    echo "==> Start Ollama in another terminal with: ollama serve"
  else
    echo "==> Ollama not found. Install it from https://ollama.com to run fully"
    echo "    offline for \$0, OR set LLM_PROVIDER=groq / openai in .env."
  fi
fi

echo ""
echo "==> Setup complete."
echo "    1. Edit .env to choose LLM_PROVIDER (ollama / groq / openai)."
echo "    2. Run the API:        uvicorn app.main:app --reload"
echo "    3. Run the UI:         streamlit run streamlit_app.py"
echo "    4. Or run everything:  docker-compose up --build"
