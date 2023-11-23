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

    b = (-a).params
    b_exp = -(a.params)
    assert b == b_exp


def test_equality():
    domain = Domain((1, ), (1.0, ))
    a = Linear(jnp.asarray([1.0]), domain)
    b = Linear(jnp.asarray([1.0]), domain)
    assert a == b

    c = Linear(jnp.asarray([2.0]), domain)
    assert a != c

    d = Linear(jnp.asarray([1.0]), Domain((2, ), (1.0, )))
    assert a != d
