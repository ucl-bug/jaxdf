#!/bin/bash
# Script to test notebooks locally
# Usage: ./scripts/test_notebooks.sh [notebook_name]
# Example: ./scripts/test_notebooks.sh quickstart.ipynb

set -e

# Get script directory and source shared library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/notebook_common.sh"

OUTPUT_DIR="/tmp/jaxdf_notebook_tests"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to test a single notebook
test_notebook() {
    local notebook=$1
    local basename=$(basename "$notebook")

    echo -e "${YELLOW}Testing: $basename${NC}"

    if run_nbconvert "$notebook" "$OUTPUT_DIR/$basename" 2>&1 | tee "$OUTPUT_DIR/${basename}.log"; then
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
    print_summary $total $failed

    if [ $failed -gt 0 ]; then
        exit 1
    fi
else
    # Test specific notebook
    notebook=$(resolve_notebook_path "$1")
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Notebook not found: $1${NC}"
        list_notebooks
        exit 1
    fi

    test_notebook "$notebook"
fi
