import jax
import numpy as np
from jax import jit
from jax import numpy as jnp

from jaxdf import *

ATOL=1e-6

domain = geometry.Domain()

# Fields on grid
x = OnGrid(jnp.asarray([1.0]), domain)
y = OnGrid(jnp.asarray([2.0]), domain)

# Continuous fields
def f(p, x):
  return jnp.expand_dims(jnp.sum(p*(x**2)), -1)
a = Continuous(5.0, domain, f)
b = Continuous(6.0, domain, f)

def test_compose_continuous():
  z = operators.compose(a)(jnp.exp)

  assert np.allclose(
    z.get_field(domain.origin), 1.
  )

def test_compose_ongrid():
  z = operators.compose(x)(jnp.exp)
  assert z.params == jnp.exp(1.0)

def test_compose_gradient():
  @jit
  def f(x):
    z = operators.compose(x)(jnp.exp)
    print(z.dims)
    return operators.gradient(z)

  print(f(a))
  print(f(a).get_field(domain.origin + 1))


if __name__ == '__main__':
  with jax.checking_leaks():
    test_compose_continuous()
    test_compose_ongrid()
    test_compose_gradient()
