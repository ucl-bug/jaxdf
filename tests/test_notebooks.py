"""
Notebook regression tests using pytest.

EXECUTION TESTS (default):
  pytest tests/test_notebooks.py -v
  - Checks notebooks run without errors
  - Runs automatically in CI

OUTPUT COMPARISON TESTS (nbval):
  pytest docs/notebooks/ --nbval
  - Compares cell outputs against stored values
  - Runs automatically in CI on pull requests
  - Use after updating outputs: ./scripts/update_notebook_outputs.sh

Run specific notebook: pytest tests/test_notebooks.py -k quickstart
Test specific notebook with regression: pytest docs/notebooks/quickstart.ipynb --nbval
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
def test_notebook_execution(notebook, tmp_path, request):
  """Execute notebook and verify it runs without errors."""
  # Apply slow marker if needed
  if notebook.name in SLOW_NOTEBOOKS:
    request.applymarker(pytest.mark.slow)

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
def test_all_notebooks_unique_names():
  """Verify that all notebook filenames are unique."""
  notebook_names = {nb.stem for nb in NOTEBOOKS}
  assert len(notebook_names) == len(
      NOTEBOOKS), "Duplicate notebook names found"
