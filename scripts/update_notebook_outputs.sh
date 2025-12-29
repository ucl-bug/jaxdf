#!/bin/bash
# Update notebook outputs to create new "golden" reference
# Run this after intentional changes to library that affect notebook outputs
# Usage: ./scripts/update_notebook_outputs.sh [notebook_name]

set -e

# Get script directory and source shared library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/notebook_common.sh"

echo -e "${YELLOW}================================================${NC}"
echo -e "${YELLOW}Notebook Output Update Tool${NC}"
echo -e "${YELLOW}================================================${NC}"
echo ""
echo "This will re-execute notebooks and update their outputs."
echo "Use this when you've made intentional changes to the library"
echo "that affect notebook results."
echo ""

# Function to update a single notebook
update_notebook() {
    local notebook=$1
    local basename=$(basename "$notebook")

    echo -e "${YELLOW}Updating: $basename${NC}"

    # Create backup
    cp "$notebook" "${notebook}.backup"
    echo "  Created backup: ${notebook}.backup"

    # Execute and update in place
    if run_nbconvert "$notebook" "$notebook" --inplace 2>&1 | grep -v "^$"; then
        echo -e "${GREEN}✓ Updated: $basename${NC}"
        rm "${notebook}.backup"
        echo "  Removed backup"
        return 0
    else
        echo -e "${RED}✗ Failed to update: $basename${NC}"
        echo "  Restoring from backup..."
        mv "${notebook}.backup" "$notebook"
        return 1
    fi
}

# Main execution
if [ $# -eq 0 ]; then
    # Update all notebooks
    echo -e "${YELLOW}Updating ALL notebooks...${NC}"
    read -p "Are you sure? This will modify all notebooks in docs/notebooks/. [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    echo ""

    failed=0
    total=0

    for notebook in "$NOTEBOOKS_DIR"/*.ipynb; do
        if [ -f "$notebook" ]; then
            ((total++))
            if ! update_notebook "$notebook"; then
                ((failed++))
            fi
            echo ""
        fi
    done

    print_summary $total $failed
    echo ""
    echo -e "${YELLOW}Important: Review the changes before committing!${NC}"
    echo "Run: git diff docs/notebooks/"

    if [ $failed -gt 0 ]; then
        exit 1
    fi
else
    # Update specific notebook
    notebook=$(resolve_notebook_path "$1")
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Notebook not found: $1${NC}"
        list_notebooks
        exit 1
    fi

    update_notebook "$notebook"
    echo ""
    echo -e "${YELLOW}Review the changes before committing:${NC}"
    echo "git diff $notebook"
fi
