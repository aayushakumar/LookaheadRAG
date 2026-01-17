#!/bin/bash
# =============================================================================
# Quick Start Script
# =============================================================================
# One-liner to get LookaheadRAG up and running
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}🚀 LookaheadRAG Quick Start${NC}"
echo ""

# Step 1: Setup (if not already done)
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}[1/4] Setting up environment...${NC}"
    ./setup.sh
else
    echo -e "${GREEN}[1/4] ✓ Environment already set up${NC}"
    source .venv/bin/activate
fi

# Step 2: Run quick tests
echo -e "${YELLOW}[2/4] Running quick tests...${NC}"
pytest tests/test_planner.py tests/test_metrics.py -v --tb=short -q 2>/dev/null || {
    echo -e "${YELLOW}  Some tests require dependencies. Installing...${NC}"
    pip install -q -e ".[dev]"
    pytest tests/test_planner.py tests/test_metrics.py -v --tb=short -q
}
echo -e "${GREEN}  ✓ Core tests passed${NC}"

# Step 3: Check Ollama
echo -e "${YELLOW}[3/4] Checking Ollama...${NC}"
if command -v ollama &> /dev/null && curl -s http://localhost:11434/api/version &> /dev/null; then
    echo -e "${GREEN}  ✓ Ollama is running${NC}"
    OLLAMA_READY=true
else
    echo -e "${YELLOW}  ! Ollama not running. Demo will use limited mode.${NC}"
    echo -e "${YELLOW}    To enable full demo: ollama serve && ollama pull llama3.2:3b${NC}"
    OLLAMA_READY=false
fi

# Step 4: Show next steps
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                       Ready to Go! 🎉                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Try these commands:${NC}"
echo ""
echo "  # Activate environment (in any new terminal)"
echo "  source .venv/bin/activate"
echo ""
echo "  # Run interactive demo"
echo "  python scripts/demo.py"
echo ""
echo "  # Run all tests"
echo "  ./run_tests.sh"
echo ""
echo "  # Download evaluation data"
echo "  python scripts/download_data.py --subset 100"
echo ""
echo "  # See all commands"
echo "  ./run.sh --help"
echo ""

if [ "$OLLAMA_READY" = false ]; then
    echo -e "${YELLOW}⚠️  To enable LLM features, run in another terminal:${NC}"
    echo "  ollama serve"
    echo "  ollama pull llama3.2:3b"
    echo ""
fi
