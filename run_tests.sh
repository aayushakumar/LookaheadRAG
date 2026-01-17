#!/bin/bash
# =============================================================================
# LookaheadRAG Test Runner
# =============================================================================
# Runs the complete test suite with coverage reporting
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  LookaheadRAG Test Suite                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Activate virtual environment if exists
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}! No virtual environment found. Run ./setup.sh first${NC}"
    echo -e "${YELLOW}  Continuing with system Python...${NC}"
fi

# Parse arguments
COVERAGE=false
VERBOSE=false
MARKERS=""
PATTERN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --unit)
            MARKERS="-m 'not slow and not integration'"
            shift
            ;;
        --integration)
            MARKERS="-m integration"
            shift
            ;;
        --slow)
            MARKERS="-m slow"
            shift
            ;;
        -k)
            PATTERN="-k $2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --coverage    Run with coverage reporting"
            echo "  -v, --verbose     Verbose output"
            echo "  --unit            Run only unit tests (fast)"
            echo "  --integration     Run only integration tests"
            echo "  --slow            Run slow tests (including VectorStore)"
            echo "  -k PATTERN        Run tests matching pattern"
            echo "  -h, --help        Show this help"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                    # Run all fast tests"
            echo "  ./run_tests.sh --coverage         # Run with coverage"
            echo "  ./run_tests.sh -k test_planner    # Run planner tests only"
            echo "  ./run_tests.sh --slow             # Run slow tests too"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Build pytest command
CMD="pytest tests/"

if [ "$VERBOSE" = true ]; then
    CMD="$CMD -v"
fi

if [ "$COVERAGE" = true ]; then
    CMD="$CMD --cov=src --cov-report=term-missing --cov-report=html"
fi

if [ -n "$MARKERS" ]; then
    CMD="$CMD $MARKERS"
else
    # Default: skip slow tests
    CMD="$CMD -m 'not slow'"
fi

if [ -n "$PATTERN" ]; then
    CMD="$CMD $PATTERN"
fi

# Add common options
CMD="$CMD --tb=short"

echo -e "${YELLOW}Running: $CMD${NC}"
echo ""

# Run tests
eval $CMD

# Show coverage report location if generated
if [ "$COVERAGE" = true ]; then
    echo ""
    echo -e "${GREEN}Coverage report generated at: htmlcov/index.html${NC}"
fi

echo ""
echo -e "${GREEN}✓ Tests completed!${NC}"
