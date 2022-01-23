from jaxdf import *
from jax import jit, make_jaxpr
import inspect
import numpy as np
from jax import numpy as jnp
import jax

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

def test_override_operator():
  z = operators.compose(x)(jnp.exp)
  assert z.params == jnp.exp(1.0)
  
  @operator
  def compose(x: OnGrid, params=Params):
    def decorator(fun):
      return x.replace_params(fun(x.params) + 100)
    return decorator, None
  
  z = operators.compose(x)(jnp.exp)
  assert z.params == jnp.exp(1.0) + 100
  
def test_jit_get_field():
  @jit
  def f(x):
    q = x + 2
    return q.get_field(domain.origin)
  
  z = f(a)
  assert np.allclose(z, [2.])
    
if __name__ == '__main__':
  with jax.checking_leaks():
    test_override_operator()
    test_jit_get_field()