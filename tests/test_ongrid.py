
import pytest
from jax import numpy as jnp

from jaxdf.discretization import FiniteDifferences, FourierSeries, OnGrid
from jaxdf.geometry import Domain


# Tests the initialization of fields OnGrid (and subclasses)
@pytest.mark.parametrize("N", [(64,), (64,64), (64, 64, 64)])
@pytest.mark.parametrize("discretization", [
  OnGrid, FourierSeries, FiniteDifferences
])
@pytest.mark.parametrize("out_dims", [0, 1, 3])
def test_create_field(N, discretization, out_dims):
    domain = Domain(N, dx=[1.]*len(N))
    true_size = list(N)
    if out_dims == 0:
      true_size += [1]
    else:
      true_size += [out_dims]

    params = jnp.ones(domain.N)
    if out_dims > 0:
      params = jnp.expand_dims(params, -1)
    if out_dims > 1:
      params = jnp.concatenate([params]*3, -1)

    field = discretization(params, domain)
    assert field.params.shape == tuple(true_size)
