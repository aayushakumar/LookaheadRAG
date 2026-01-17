#!/bin/bash
# =============================================================================
# LookaheadRAG Setup Script
# =============================================================================
# This script sets up the complete development environment including:
# - Python virtual environment
# - All dependencies
# - Ollama LLM (optional)
# - Sample data download
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              LookaheadRAG Environment Setup                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Check Python version
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/6] Checking Python version...${NC}"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
        echo -e "${GREEN}  ✓ Python $PYTHON_VERSION found${NC}"
    else
        echo -e "${RED}  ✗ Python 3.9+ required, found $PYTHON_VERSION${NC}"
        exit 1
    fi
else
    echo -e "${RED}  ✗ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 2: Create virtual environment
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/6] Creating virtual environment...${NC}"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}  ! Virtual environment already exists at $VENV_DIR${NC}"
    read -p "  Recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
        echo -e "${GREEN}  ✓ Virtual environment recreated${NC}"
    else
        echo -e "${GREEN}  ✓ Using existing virtual environment${NC}"
    fi
else
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}  ✓ Virtual environment created at $VENV_DIR${NC}"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}  ✓ Virtual environment activated${NC}"

# -----------------------------------------------------------------------------
# Step 3: Upgrade pip and install dependencies
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/6] Installing dependencies...${NC}"

pip install --quiet --upgrade pip wheel setuptools

# Install project in editable mode with all extras
pip install --quiet -e ".[all]"

echo -e "${GREEN}  ✓ All dependencies installed${NC}"

# -----------------------------------------------------------------------------
# Step 4: Check/Install Ollama (optional)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/6] Checking Ollama (for local LLM)...${NC}"

if command -v ollama &> /dev/null; then
    echo -e "${GREEN}  ✓ Ollama is installed${NC}"
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/version &> /dev/null; then
        echo -e "${GREEN}  ✓ Ollama server is running${NC}"
    else
        echo -e "${YELLOW}  ! Ollama is not running. Start it with: ollama serve${NC}"
    fi
    
    # Check for required models
    echo -e "${YELLOW}  Checking for required models...${NC}"
    
    if ollama list 2>/dev/null | grep -q "llama3.2:3b"; then
        echo -e "${GREEN}  ✓ llama3.2:3b is available${NC}"
    else
        echo -e "${YELLOW}  ! llama3.2:3b not found. Pull with: ollama pull llama3.2:3b${NC}"
    fi
    
    if ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
        echo -e "${GREEN}  ✓ llama3.1:8b is available${NC}"
    else
        echo -e "${YELLOW}  ! llama3.1:8b not found (optional). Pull with: ollama pull llama3.1:8b${NC}"
    fi
else
    echo -e "${YELLOW}  ! Ollama not found. Install with: brew install ollama${NC}"
    echo -e "${YELLOW}    Or use Groq API (free) instead - set GROQ_API_KEY in .env${NC}"
fi

# -----------------------------------------------------------------------------
# Step 5: Create .env file if not exists
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[5/6] Setting up configuration...${NC}"

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    echo -e "${GREEN}  ✓ Created .env from template${NC}"
    echo -e "${YELLOW}    Edit .env to add your API keys (optional)${NC}"
else
    echo -e "${GREEN}  ✓ .env already exists${NC}"
fi

# Create data and logs directories
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/results"
echo -e "${GREEN}  ✓ Created data/, logs/, results/ directories${NC}"

# -----------------------------------------------------------------------------
# Step 6: Verify installation
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[6/6] Verifying installation...${NC}"

python -c "import src; print(f'  ✓ LookaheadRAG v{src.__version__} installed')"
python -c "import chromadb; print('  ✓ ChromaDB installed')"
python -c "import sentence_transformers; print('  ✓ Sentence Transformers installed')"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Setup Complete! 🎉                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Quick Start:${NC}"
echo ""
echo "  # Activate the environment"
echo "  source .venv/bin/activate"
echo ""
echo "  # Run tests"
echo "  ./run_tests.sh"
echo ""
echo "  # Run demo"
echo "  python scripts/demo.py"
echo ""
echo "  # Full help"
echo "  ./run.sh --help"
echo ""
