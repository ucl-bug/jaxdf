from re import A
from jaxdf import *
from jax import jit, make_jaxpr, grad
from jax import numpy as jnp
import inspect
import jax


domain = geometry.Domain()

# Fields on grid
x = OnGrid(1.0, domain)
y = OnGrid(2.0, domain)

def f(p, x):
  return p + x
a = Continuous(5.0, domain, f)
b = Continuous(6.0, domain, f)

def test_paramfun():
  a = operators.dummy(x)
  
def test_jit_paramfun():
  @jit
  def f(x):
    return operators.dummy(x)
  _ = f(x)

def test_get_params():
  op_params = operators.dummy.get_params(x)
  assert op_params['k'] == 3
  
  def f(x, op_params):
    return operators.dummy(x, params=op_params)
  
  z = f(x, op_params)
  assert z.params == 3.0
  
  z = jit(f)(x, op_params)
  assert z.params == 3.0
  
  op_params = operators.dummy.get_params(a)
  z = jit(f)(a, op_params)
  print(z)
  
  def f(x, coord, op_params):
    b = operators.dummy(x, params=op_params)
    return b.get_field(coord)
  
  z = jit(f)(a, 1.0, op_params)
  print(make_jaxpr(f)(a, 1.0, op_params))
  print(z)
  
def test_grad():
  def loss(x, y):
    z = x**2 + y * 5 + x*y
    return jnp.sum(z.params)
  
  gradfn = grad(loss, argnums=(0, 1))
  x_grad, y_grad = gradfn(x, y)
  print(x_grad)
  assert x_grad.params == 4.0
  assert y_grad.params == 6.0
  
if __name__ == '__main__':
  with jax.checking_leaks():
    test_paramfun()
    test_jit_paramfun()
    test_get_params()
    test_grad()