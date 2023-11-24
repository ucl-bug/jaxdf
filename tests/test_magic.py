from equinox.internal._omega import ω
from jax import numpy as jnp

from jaxdf import OnGrid, geometry


def test_equinox_omega_wrapper():
  domain = geometry.Domain((1, ), (1.0, ))
  a = OnGrid(jnp.asarray([1.0]), domain)

  b = a**ω
  c = ω(a)

  assert type(b) == type(c)
