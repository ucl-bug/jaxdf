import json
import os
from typing import List

import numpy as np
import pytest
from jax import numpy as jnp

from jaxdf import FourierSeries
from jaxdf.geometry import Domain
from jaxdf.operators import gradient

TEST_DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + "/test_data/"


def get_filenames(incipit: str,
                  base_folder=TEST_DATA_PATH,
                  extension=".json") -> List[str]:
  # Lists all the files in `base_folder` starting with `incipit`
  all_files = os.listdir(base_folder)
  good_files = [
      base_folder + f for f in all_files
      if f.startswith(incipit) and f.endswith(extension)
  ]
  return good_files


def load_test_data(filename):
  with open(filename, "r") as f:
    data = json.load(f)

  # Parsing data
  if data["is_complex"]:
    data["x"] = np.asarray(data["x"][0]) + 1j * np.asarray(data["x"][1])
    data["y"] = np.asarray(data["y"][0]) + 1j * np.asarray(data["y"][1])
  else:
    data["x"] = np.asarray(data["x"][0])
    data["y"] = np.asarray(data["y"][0])

  if type(data["step_size"]) in [int, float]:
    data["step_size"] = [float(data["step_size"])]

  else:
    data["step_size"] = list(map(float, data["step_size"]))

  return data


FILENAMES = get_filenames("gradient_FourierSeries_")


@pytest.mark.parametrize("filename", FILENAMES)
def test_gradient_accuracy_fourier_series(
    filename,
    make_plot=False,
):
  # Load data in a ditionary
  test_data = load_test_data(filename)

  # Making fields
  domain = Domain(test_data["x"].shape, test_data["step_size"])
  x = FourierSeries.from_grid(test_data["x"], domain)

  # Getting gradient
  y_pred = gradient(x)

  # Checking if similar
  if len(test_data["y"].shape) == len(x.domain.N):
    y = jnp.expand_dims(test_data["y"], -1)
  else:
    y = test_data["y"]
  norm = jnp.linalg.norm(y)
  relErr = jnp.amax(jnp.abs(y_pred.on_grid - y)) / norm

  # Check that the maximum error is smaller than 0.01%
  print("\n|")
  print("- Size: ", y.shape)
  print("- Stepsize: ", test_data["step_size"])
  print("- Complex: ", test_data["is_complex"])
  print("- Relative max error = ", 100 * relErr, "%")

  # Plot fft of the two signals
  if make_plot:
    from matplotlib import pyplot as plt

    if len(test_data["step_size"]) == 1:
      f1 = jnp.fft.fftn(y[..., 0])
      f2 = jnp.fft.fftn(y_pred.on_grid[..., 0])
      plt.plot(jnp.abs(f1), label="true spectrum")
      plt.plot(jnp.abs(f2), label="predicted spectrum")
      plt.legend()
      plt.savefig(TEST_DATA_PATH + "/last_test.png")
      plt.close()

  assert relErr < 1e-4
