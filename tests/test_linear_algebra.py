from jax import numpy as jnp

from jaxdf import Domain, OnGrid
from jaxdf.operators.linear_algebra import dot_product


def test_dot_product_ongrid():
  domain = Domain((3, 3), (0.5, 0.5))
  params_1 = jnp.ones((3, 3, 1)) * 2.0
  params_2 = jnp.ones((3, 3, 1)) * 3.0

  x = OnGrid(params_1, domain)
  y = OnGrid(params_2, domain)

  z = dot_product(x, y)
  assert z == 54.0
