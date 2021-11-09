from jaxdf.geometry import Domain
from jaxdf.discretization import Arbitrary
from jax.random import normal, PRNGKey
from jax import numpy as jnp

# 1D tests
def make_polynomial():
    domain = Domain((16,),(2/16.,))
    N = 10

    def p_n(theta, x):
        i = jnp.arange(N)
        return jnp.sum(theta*(x**i))

    def init_params(seed, domain):
        return normal(seed, (N,))

    polynomial_discretization = Arbitrary(domain, p_n, init_params)
    return polynomial_discretization

def test_polynomial_params():
    polynomial_discretization = make_polynomial()
    y_params, y = polynomial_discretization.random_field(seed=PRNGKey(0), name='y')
    assert jnp.allclose(
        y_params,
        jnp.array([-0.372111  ,  0.26423106, -0.18252774, -0.7368198 ,
             -0.44030386, -0.15214427, -0.6713536 , -0.5908642 ,
              0.73168874,  0.5673025 ], dtype=jnp.float32)
    )

def test_polynomial_grid_values():
    polynomial_discretization = make_polynomial()
    y_params, y = polynomial_discretization.random_field(seed=PRNGKey(0), name='y')
    y_sampled = polynomial_discretization.get_field_on_grid()(y_params)
    assert jnp.allclose(
        y_sampled, 
        jnp.array([-0.36354282, -0.45370287, -0.48655397, -0.48947105,
             -0.47688618, -0.4541707 , -0.42374668, -0.38916516,
             -0.35649633, -0.3344538 , -0.33521673, -0.3769234 ,
             -0.48628226, -0.6956897 , -1.0236492 , -1.420161  ], dtype=jnp.float32))

def test_polynomial_field_values():
    polynomial_discretization = make_polynomial()
    y_params, y = polynomial_discretization.random_field(seed=PRNGKey(0), name='y')
    y_sample = polynomial_discretization.get_field()(y_params, 0.75)
    assert jnp.allclose(y_sample, -0.84538496)