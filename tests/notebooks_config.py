"""
Configuration for notebook testing.

This file defines which notebooks should be excluded from testing.
It's used by both pytest (tests/test_notebooks.py) and GitHub Actions
(.github/workflows/notebook_tests.yml) to ensure consistency.
"""

# Notebooks to exclude from all testing (pytest and CI)
# These are computationally heavy notebooks that have unit tests
# extracted to tests/test_notebook_examples.py instead
EXCLUDED_NOTEBOOKS = {
    "example_1_paper.ipynb",
    "helmholtz_pinn.ipynb",
    "simulate_helmholtz_equation.ipynb",
    "pinn_burgers.ipynb",
}

# Notebooks that take >60 seconds to execute
SLOW_NOTEBOOKS = set()    # All heavy notebooks now excluded
