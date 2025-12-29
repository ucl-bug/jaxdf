import sys

import pytest

from tests.notebooks_config import EXCLUDED_NOTEBOOKS


# each test runs on cwd to its temp dir
@pytest.fixture(autouse=True)
def go_to_tmpdir(request):
  # Get the fixture dynamically by its name.
  tmpdir = request.getfixturevalue("tmpdir")
  # ensure local test created packages can be imported
  sys.path.insert(0, str(tmpdir))
  # Chdir only for the duration of the test.
  with tmpdir.as_cwd():
    yield


def pytest_ignore_collect(collection_path, config):
  """Ignore notebooks listed in EXCLUDED_NOTEBOOKS during collection."""
  if collection_path.suffix == ".ipynb":
    if collection_path.name in EXCLUDED_NOTEBOOKS:
      return True
  return False
