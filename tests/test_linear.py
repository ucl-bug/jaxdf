import pytest
from jax import numpy as jnp

from jaxdf.discretization import Linear
from jaxdf.geometry import Domain


@pytest.mark.parametrize("N", [(33, ), (33, 33), (33, 33, 33)])
def test_create(N):
    domain = Domain(N, dx=[1.0] * len(N))
    params = jnp.zeros(N)
    field = Linear(params, domain)
    assert field.params.shape == N
