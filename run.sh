#!/bin/bash
# =============================================================================
# LookaheadRAG Main Runner
# =============================================================================
# Main script to run various LookaheadRAG commands
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

# Activate virtual environment if exists
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

show_help() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                     LookaheadRAG Runner                        ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Usage:${NC} ./run.sh <command> [options]"
    echo ""
    echo -e "${CYAN}Commands:${NC}"
    echo "  setup           Set up the development environment"
    echo "  test            Run the test suite"
    echo "  demo            Run interactive demo"
    echo "  download-data   Download HotpotQA dataset"
    echo "  build-index     Build vector index from data"
    echo "  evaluate        Run full evaluation"
    echo "  lint            Run linter (ruff)"
    echo "  format          Format code (ruff)"
    echo "  typecheck       Run type checker (mypy)"
    echo "  clean           Clean build artifacts"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo "  ./run.sh setup                    # Set up environment"
    echo "  ./run.sh test --coverage          # Run tests with coverage"
    echo "  ./run.sh demo                     # Interactive demo"
    echo "  ./run.sh evaluate --subset 100    # Evaluate on 100 examples"
    echo ""
}

cmd_setup() {
    ./setup.sh "$@"
}

cmd_test() {
    ./run_tests.sh "$@"
}

cmd_demo() {
    python scripts/demo.py "$@"
}

cmd_download_data() {
    echo -e "${BLUE}Downloading HotpotQA dataset...${NC}"
    python scripts/download_data.py "$@"
}

cmd_build_index() {
    echo -e "${BLUE}Building vector index...${NC}"
    python scripts/build_index.py "$@"
}

cmd_evaluate() {
    echo -e "${BLUE}Running evaluation...${NC}"
    python scripts/run_evaluation.py "$@"
}

cmd_lint() {
    echo -e "${BLUE}Running linter...${NC}"
    ruff check src/ tests/ eval/ scripts/
}

cmd_format() {
    echo -e "${BLUE}Formatting code...${NC}"
    ruff format src/ tests/ eval/ scripts/
    ruff check --fix src/ tests/ eval/ scripts/
}

cmd_typecheck() {
    echo -e "${BLUE}Running type checker...${NC}"
    mypy src/ --ignore-missing-imports
}

cmd_clean() {
    echo -e "${BLUE}Cleaning build artifacts...${NC}"
    rm -rf build/ dist/ *.egg-info/
    rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
    rm -rf htmlcov/ .coverage
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Cleaned!${NC}"
}

# Parse command
COMMAND="${1:-help}"
shift || true

case $COMMAND in
    setup)
        cmd_setup "$@"
        ;;
    test)
        cmd_test "$@"
        ;;
    demo)
        cmd_demo "$@"
        ;;
    download-data|download)
        cmd_download_data "$@"
        ;;
    build-index|index)
        cmd_build_index "$@"
        ;;
    evaluate|eval)
        cmd_evaluate "$@"
        ;;
    lint)
        cmd_lint "$@"
        ;;
    format|fmt)
        cmd_format "$@"
        ;;
    typecheck|types)
        cmd_typecheck "$@"
        ;;
    clean)
        cmd_clean "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
