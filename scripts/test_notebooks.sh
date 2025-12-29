#!/bin/bash
# Script to test notebooks locally
# Usage: ./scripts/test_notebooks.sh [notebook_name]
# Example: ./scripts/test_notebooks.sh quickstart.ipynb

set -e

NOTEBOOKS_DIR="docs/notebooks"
OUTPUT_DIR="/tmp/jaxdf_notebook_tests"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to test a single notebook
test_notebook() {
    local notebook=$1
    local basename=$(basename "$notebook")

    echo -e "${YELLOW}Testing: $basename${NC}"

    if JAX_PLATFORMS=cpu jupyter nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.timeout=600 \
        --ExecutePreprocessor.kernel_name=python3 \
        --output "$OUTPUT_DIR/$basename" \
        "$notebook" 2>&1 | tee "$OUTPUT_DIR/${basename}.log"; then
        echo -e "${GREEN}✓ PASSED: $basename${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED: $basename${NC}"
        echo "  Log: $OUTPUT_DIR/${basename}.log"
        return 1
    fi
}

# Main execution
if [ $# -eq 0 ]; then
    # Test all notebooks
    echo "Testing all notebooks in $NOTEBOOKS_DIR..."
    echo "Output directory: $OUTPUT_DIR"
    echo ""

    failed=0
    total=0

    for notebook in "$NOTEBOOKS_DIR"/*.ipynb; do
        if [ -f "$notebook" ]; then
            ((total++))
            if ! test_notebook "$notebook"; then
                ((failed++))
            fi
            echo ""
        fi
    done

    # Summary
    echo "=================================="
    echo "Summary:"
    echo "  Total: $total"
    echo "  Passed: $((total - failed))"
    echo "  Failed: $failed"
    echo "=================================="

    if [ $failed -gt 0 ]; then
        exit 1
    fi
else
    # Test specific notebook
    notebook="$NOTEBOOKS_DIR/$1"
    if [ ! -f "$notebook" ]; then
        # Try without directory prefix
        notebook="$1"
        if [ ! -f "$notebook" ]; then
            echo -e "${RED}Error: Notebook not found: $1${NC}"
            echo "Available notebooks:"
            ls -1 "$NOTEBOOKS_DIR"/*.ipynb
            exit 1
        fi
    fi

    test_notebook "$notebook"
fi
