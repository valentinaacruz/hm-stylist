#!/usr/bin/env bash
# Arranca HyM. Uso:  ./run.sh   (desde cualquier carpeta)
# Cambia las variables aquí o al llamar, p. ej.:  LLM_MODEL=Gemma-4-E4B-it-GGUF ./run.sh
cd "$(dirname "$0")"
export HM_DATA_DIR="${HM_DATA_DIR:-$HOME/hm-demo-data}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:13305/v1}"   # Lemonade (Ollama: http://localhost:11434/v1)
export LLM_MODEL="${LLM_MODEL:-Qwen3.6}"
export LLM_THINKING="${LLM_THINKING:-0}"                            # Qwen3.x sin razonamiento: más rápido
export LLM_VISION="${LLM_VISION:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"                       # 1 = sin Internet (Fashion-CLIP ya descargado)
[ -f ../.venv/bin/activate ] && source ../.venv/bin/activate
echo "Datos: $HM_DATA_DIR · LLM: $LLM_MODEL @ $LLM_BASE_URL"
exec python -m uvicorn server:app --host 0.0.0.0 --port 8000
