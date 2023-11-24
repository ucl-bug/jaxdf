from jax import numpy as jnp

from jaxdf.core import operator
from jaxdf.discretization import OnGrid

from .functions import compose


@operator.abstract
def dot_product(x, y):    # pragma: no cover
  raise NotImplementedError


@operator
def dot_product(x: OnGrid, y: OnGrid, *, params=None):
  r"""Computes the dot product of two fields."""
  x_conj = compose(x)(jnp.conj)
  return jnp.sum((x_conj * y).on_grid)
