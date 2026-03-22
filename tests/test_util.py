from jax import numpy as jnp

from jaxdf import util
from jaxdf.discretization import Continuous, FiniteDifferences, FourierSeries
from jaxdf.operators.differential import gradient
from jaxdf.util import get_implementations, has_implementation


def test_append_dimension():
  a = jnp.zeros((2, 3))
  b = util.append_dimension(a)
  assert b.shape == (2, 3, 1)


def test_update_dictionary():
  a = {"a": 1, "b": 2}
  b = {"b": 3, "c": 4}
  c = util.update_dictionary(a, b)
  assert c == {"a": 1, "b": 3, "c": 4}


def test_get_implemented():
  util.get_implemented(gradient)


def test_get_implementations():
  impls = get_implementations(gradient)
  assert isinstance(impls, list)
  assert len(impls) >= 3    # At least Continuous, FD, Fourier
  assert ('FourierSeries', ) in impls
  assert ('FiniteDifferences', ) in impls
  assert ('Continuous', ) in impls


def test_has_implementation():
  assert has_implementation(gradient, FourierSeries) == True
  assert has_implementation(gradient, FiniteDifferences) == True
  assert has_implementation(gradient, Continuous) == True


if __name__ == "__main__":
  test_get_implemented()
