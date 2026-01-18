#!/bin/bash
# Run the LookaheadRAG web server

set -e

# Change to project directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Install web dependencies if needed
pip install -q fastapi uvicorn[standard] 2>/dev/null || true

# Export environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     LookaheadRAG Web UI                      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Open in browser: http://localhost:8000                      ║"
echo "║  API docs:        http://localhost:8000/docs                 ║"
echo "║  Press Ctrl+C to stop                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Run the server
cd web
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
