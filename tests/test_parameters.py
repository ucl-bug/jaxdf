from re import A
from jaxdf import *
from jax import jit, make_jaxpr, grad
from jax import numpy as jnp
import inspect

# Fields on grid
x = OnGrid(1.0, 'domain')
y = OnGrid(2.0, 'domain')

def test_paramfun():
  a = operators.dummy(x)
  
def test_jit_paramfun():
  @jit
  def f(x):
    return operators.dummy(x)
  _ = f(x)

def test_get_params():
  op_params = operators.dummy.get_params(x)
  assert op_params.params['k'] == 3
  
  def f(x, op_params):
    return operators.dummy(x, params=op_params)
  
  z = f(x, op_params)
  assert z.params == 3.0
  
  z = jit(f)(x, op_params)
  assert z.params == 3.0
  
def test_grad():
  def loss(x, y):
    z = x**2 + y * 5 + x*y
    return jnp.sum(z.params)
  
  gradfn = grad(loss, argnums=(0, 1))
  x_grad, y_grad = gradfn(x, y)
  assert x_grad.params == 4.0
  assert y_grad.params == 6.0
  
if __name__ == '__main__':
  test_paramfun()
  test_jit_paramfun()
  test_get_params()
  test_grad()