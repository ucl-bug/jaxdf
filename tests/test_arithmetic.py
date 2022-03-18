import jax
import numpy as np
from jax import jit

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

def test_add():
  z = x + y
  assert z.params == 3.0
  assert type(z) == OnGrid

  z = x + 5.
  assert z.params == 6.0

def test_add_continuous():
  z = a + b
  z_val = z.get_field(domain.origin + 1)
  print(z_val)
  assert np.allclose(z_val, [22.])

def test_jit_continuous():
  @jit
  def f(a, b):
    return a + b

  z = f(a,b)

def test_sub():
  z = x - y
  assert z.params == -1.0
  assert type(z) == OnGrid

  z = x - 2.0
  assert z.params == -1.0

def test_jit():

  @jit
  def prod(x, y):
    return x + y
  return prod(x, y)

def test_jit_with_float():
  @jit
  def add(x, y):
    return x + y * 10

  _ = add(x,y)
  _ = add(x, 6.0)
  _ = add(-5.0, x)
  _ = add(a,b)

if __name__ == '__main__':
  with jax.checking_leaks():
    test_add()
    test_sub()
    test_jit()
    test_jit_with_float()
    test_add_continuous()
    test_jit_continuous()
