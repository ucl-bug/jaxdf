from jaxdf.core import operator
from jaxdf.geometry import Domain
from jaxdf.discretization import Arbitrary, FourierSeries
from jax.random import normal, PRNGKey, split
from jax import numpy as jnp

# 1D Example arbitrary
domain = Domain((16,),(2/16.,))
N = 10
coord_value = 0.1

def p_n(theta, x):
    i = jnp.arange(N)
    return jnp.sum(theta*(x**i))

def init_params(seed, domain):
    return normal(seed, (N,))

polynomial_discretization = Arbitrary(domain, p_n, init_params)
y_params, y = polynomial_discretization.random_field(seed=PRNGKey(0), name='y')
x_params, x = polynomial_discretization.random_field(seed=PRNGKey(1), name='x')

def test_sum():
    # Without operators
    f1 = polynomial_discretization.get_field()(y_params, coord_value)
    f2 = polynomial_discretization.get_field()(x_params, coord_value)
    true_val = f1 + f2

    # With operators
    @operator()
    def add(u, v):
        return u + v

    num_op = add(u=x, v=y)
    gp = num_op.get_global_params()
    pred_value = num_op.get_field(0)(gp, {"u": x_params, "v": y_params}, coord_value)
    assert jnp.allclose(true_val, pred_value)

def test_diff():
    # Without operators
    f1 = polynomial_discretization.get_field()(y_params, coord_value)
    f2 = polynomial_discretization.get_field()(x_params, coord_value)
    true_val = f1 - f2

    # With operators
    @operator()
    def diff(u, v):
        return v - u

    num_op = diff(u=x, v=y)
    gp = num_op.get_global_params()
    pred_value = num_op.get_field(0)(gp, {"u": x_params, "v": y_params}, coord_value)
    assert jnp.allclose(true_val, pred_value)

def test_prod():
    # Without operators
    f1 = polynomial_discretization.get_field()(y_params, coord_value)
    f2 = polynomial_discretization.get_field()(x_params, coord_value)
    true_val = f1 * f2

    # With operators
    @operator()
    def prod(u, v):
        return u * v

    num_op = prod(u=x, v=y)
    gp = num_op.get_global_params()
    pred_value = num_op.get_field(0)(gp, {"u": x_params, "v": y_params}, coord_value)
    assert jnp.allclose(true_val, pred_value)

def test_div():
    # Without operators
    f1 = polynomial_discretization.get_field()(y_params, coord_value)
    f2 = polynomial_discretization.get_field()(x_params, coord_value)
    true_val = f1 / f2

    # With operators
    @operator()
    def div(u, v):
        return v / u

    num_op = div(u=x, v=y)
    gp = num_op.get_global_params()
    pred_value = num_op.get_field(0)(gp, {"u": x_params, "v": y_params}, coord_value)
    assert jnp.allclose(true_val, pred_value)

if __name__ == '__main__':
    test_sum()
    test_diff()
    test_prod()