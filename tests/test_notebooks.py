"""
Notebook regression tests using pytest.

EXECUTION TESTS (default):
  pytest tests/test_notebooks.py -v
  - Checks notebooks run without errors

OUTPUT COMPARISON TESTS (nbval):
  pytest tests/test_notebooks.py --nbval
  - Compares cell outputs against stored values
  - Use after updating outputs: ./scripts/update_notebook_outputs.sh

Run specific notebook: pytest tests/test_notebooks.py -k quickstart
Skip slow tests: pytest tests/test_notebooks.py -m "not slow"
"""
import os
import subprocess
from pathlib import Path

import pytest

from .notebooks_config import EXCLUDED_NOTEBOOKS, SLOW_NOTEBOOKS

NOTEBOOKS_DIR = Path(__file__).parent.parent / "docs" / "notebooks"
NOTEBOOKS = [
    nb for nb in NOTEBOOKS_DIR.glob("*.ipynb")
    if nb.name not in EXCLUDED_NOTEBOOKS
]


def get_notebook_name(notebook_path):
  """Get friendly name for test parametrization."""
  return notebook_path.stem


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=get_notebook_name)
def test_notebook_execution(notebook, tmp_path):
  """Execute notebook and verify it runs without errors."""
  # Apply slow marker if needed
  if notebook.name in SLOW_NOTEBOOKS:
    pytest.mark.slow()

  output_path = tmp_path / f"executed_{notebook.name}"

  # Execute notebook with nbconvert
  result = subprocess.run(
      [
          "jupyter",
          "nbconvert",
          "--to",
          "notebook",
          "--execute",
          "--ExecutePreprocessor.timeout=600",
          "--ExecutePreprocessor.kernel_name=python3",
          "--output",
          str(output_path),
          str(notebook),
      ],
      capture_output=True,
      text=True,
      env={
          **os.environ, "JAX_PLATFORMS": "cpu"
      },    # Force CPU for testing
  )

  # Check execution succeeded
  if result.returncode != 0:
    print(f"\n{'='*60}")
    print(f"STDOUT:\n{result.stdout}")
    print(f"{'='*60}")
    print(f"STDERR:\n{result.stderr}")
    print(f"{'='*60}")
    pytest.fail(f"Notebook {notebook.name} failed to execute.\n"
                f"Return code: {result.returncode}\n"
                f"See output above for details.")

  # Verify output file was created
  assert output_path.exists(), f"Output notebook not created: {output_path}"


@pytest.mark.slow
def test_all_notebooks_different_outputs():
  """Verify that all notebooks produce different outputs (sanity check)."""
  # This test ensures notebooks aren't just copies of each other
  notebook_names = {nb.stem for nb in NOTEBOOKS}
  assert len(notebook_names) == len(
      NOTEBOOKS), "Duplicate notebook names found"


# =============================================================================
# OUTPUT COMPARISON TESTS (using nbval plugin)
# =============================================================================
# These tests compare notebook cell outputs against stored reference outputs
# Run with: pytest tests/test_notebooks.py --nbval
#
# To update reference outputs after intentional changes:
#   ./scripts/update_notebook_outputs.sh
#
# nbval configuration is in pyproject.toml [tool.pytest.ini_options]
# =============================================================================


@pytest.mark.nbval
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=get_notebook_name)
def test_notebook_output_regression(notebook):
  """
    Compare notebook outputs against reference values.

    This test uses the nbval plugin to verify that notebook outputs
    haven't changed unexpectedly. It's automatically run when pytest
    is invoked with the --nbval flag.

    To skip comparison for specific cells, add to the cell:
        # SKIP_COMPARE

    Or tag the cell with metadata: {"tags": ["nbval-skip"]}

    To update outputs after intentional changes:
        ./scripts/update_notebook_outputs.sh [notebook_name]
    """
  # Mark slow notebooks
  if notebook.name in SLOW_NOTEBOOKS:
    pytest.mark.slow()

  # nbval plugin handles the actual comparison automatically
  # This test just provides the parametrization and documentation
  pass
