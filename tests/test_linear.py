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


def test_op_neg():
    domain = Domain((1, ), (1.0, ))
    a = Linear(jnp.asarray([1.0]), domain)

    b = (-a).on_grid
    b_exp = -(a.on_grid)
    assert b == b_exp
