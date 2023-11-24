import jax
import numpy as np
from jax import jit
from jax import numpy as jnp

from jaxdf import *

ATOL = 1e-6

domain = geometry.Domain()

# Fields on grid
x = OnGrid(jnp.asarray([1.0]), domain)
y = OnGrid(jnp.asarray([2.0]), domain)


# Continuous fields
def f(p, x):
  return jnp.expand_dims(jnp.sum(p * (x**2)), -1)


a = Continuous(5.0, domain, f)
b = Continuous(6.0, domain, f)


def test_compose_continuous():
  z = operators.compose(a)(jnp.exp)
  assert np.allclose(z(domain.origin), 1.0)


def test_compose_ongrid():
  x = OnGrid(jnp.asarray([1.0]), domain)
  z = operators.compose(x)(jnp.exp)
  assert z.params == jnp.exp(1.0)


def test_compose_gradient():

  @jit
  def f(x):
    z = operators.compose(x)(jnp.exp)
    print(z.dims)
    return operators.gradient(z)

  print(f(a))
  print(f(a)(domain.origin + 1))


def test_compose():
  x = 2.0
  y = jnp.tanh(x)
  z = operators.compose(x)(jnp.tanh)
  assert np.allclose(z, y)


def test_functional():
  x = 2.0
  y = jnp.sum(x)
  z = operators.functions.functional(x)(jnp.sum)
  assert np.allclose(z, y)


def test_functional_ongrid():
  x = OnGrid(jnp.asarray([1.0]), domain)
  y = jnp.sum(x.on_grid)
  z = operators.functions.functional(x)(jnp.sum)
  assert np.allclose(z, y)


def test_fd_shift_operator():
  domain = geometry.Domain((5, 5), (0.5, 0.5))
  params = jnp.zeros((5, 5, 1))
  params = params.at[2, 2].set(1.0)
  x = FiniteDifferences(params, domain, accuracy=2)

  y = operators.shift_operator(x, dx=[0.125, 0.0]).on_grid[..., 0]
  y_true = jnp.asarray([
      [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
      ],
      [
          0.0,
          0.0,
          0.25,
          0.0,
          0.0,
      ],
      [
          0.0,
          0.0,
          0.75,
          0.0,
          0.0,
      ],
      [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
      ],
      [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
      ],
  ])
  assert np.allclose(y, y_true, atol=ATOL)


def test_fourier_shift_operator():
  domain = geometry.Domain((5, 5), (0.5, 0.5))
  params = jnp.zeros((5, 5, 1))
  params = params.at[2, 2].set(1.0)
  x = FourierSeries(params, domain)

  y = operators.shift_operator(x, dx=[0.125, 0.0]).on_grid[..., 0]
  y_true = jnp.asarray([
      [0.0, 0.0, -0.15872093, 0.0, 0.0],
      [0.0, 0.0, 0.3115073, 0.0, 0.0],
      [0.0, 0.0, 0.9040295, 0.0, 0.0],
      [0.0, 0.0, -0.20000003, 0.0, 0.0],
      [0.0, 0.0, 0.14318419, 0.0, 0.0],
  ])
  assert np.allclose(y, y_true, atol=ATOL)


def test_jax_leaks():
  with jax.checking_leaks():
    test_compose_continuous()
    test_compose_ongrid()
    test_compose_gradient()


if __name__ == "__main__":
  test_fourier_shift_operator()
