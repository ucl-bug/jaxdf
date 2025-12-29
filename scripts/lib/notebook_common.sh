#!/bin/bash
# Shared utilities for notebook testing and updating scripts

# Color definitions for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default paths
NOTEBOOKS_DIR="${NOTEBOOKS_DIR:-docs/notebooks}"

# Execute jupyter nbconvert with standard parameters
# Usage: run_nbconvert <notebook_path> <output_path> [extra_args...]
run_nbconvert() {
  local notebook="$1"
  local output="$2"
  shift 2
  local extra_args="$@"

  JAX_PLATFORMS=cpu jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=600 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output "$output" \
    $extra_args \
    "$notebook"
}

# Resolve notebook path
# Usage: resolve_notebook_path <notebook_name_or_path>
resolve_notebook_path() {
  local notebook="$1"

  if [ -f "$notebook" ]; then
    echo "$notebook"
    return 0
  fi

  # Try with directory prefix
  local with_dir="$NOTEBOOKS_DIR/$notebook"
  if [ -f "$with_dir" ]; then
    echo "$with_dir"
    return 0
  fi

  return 1
}

# Print summary of test results
# Usage: print_summary <total> <failed>
print_summary() {
  local total=$1
  local failed=$2
  local passed=$((total - failed))

  echo "=================================="
  echo "Summary:"
  echo "  Total: $total"
  echo "  Passed: $passed"
  echo "  Failed: $failed"
  echo "=================================="
}

# List available notebooks
list_notebooks() {
  echo "Available notebooks:"
  ls -1 "$NOTEBOOKS_DIR"/*.ipynb 2>/dev/null || echo "No notebooks found in $NOTEBOOKS_DIR"
}
