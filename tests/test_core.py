import jax
import numpy as np
from jax import jit, make_jaxpr
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

# TODO: The test below should be run, however it makes the other
#       tests on the `compose` operator fail.
'''
def test_override_operator():
  z = operators.compose(x)(jnp.exp)
  assert z.params == jnp.exp(1.0)

  @operator
  def compose(x: OnGrid, params=None):
    def decorator(fun):
      return x.replace_params(fun(x.params) + 100)
    return decorator, None

  z = operators.compose(x)(jnp.exp)
  assert z.params == jnp.exp(1.0) + 100
'''

def test_jit_get_field():
  @jit
  def f(x):
    q = x + 2
    return q.get_field(domain.origin)

  z = f(a)
  assert np.allclose(z, [2.])

def test_call_field():
  z = a(domain.origin)

  @jit
  def f(u):
    return u(domain.origin)

  _ = f(a)
  print(make_jaxpr(f)(a))

def test_make_fourier_inside_jitted_fun():
  @jax.jit
  def f(x):
    y = FourierSeries(x.params**2, x.domain)
    return y + 1

  u = FourierSeries(2., domain)  # u.params == theta
  v = f(u)
  assert v.params == 5.


if __name__ == '__main__':
  with jax.checking_leaks():
    test_call_field()
    # test_override_operator()
    test_jit_get_field()
