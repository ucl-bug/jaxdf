from jax import numpy as jnp

from jaxdf import util
from jaxdf.operators.differential import gradient


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


if __name__ == "__main__":
  test_get_implemented()
