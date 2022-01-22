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
  return p*(x**2)
a = Continuous(1.0, domain, f)
b = Continuous(6.0, domain, f)

  
def test_continous_gradient():
  z = operators.gradient(a)
  w = operators.gradient(z)
  print(z)
  print(a.get_field(3.), z.get_field(3.),w.get_field(3.))
  
def test_jit_continous_gradient():
  
  @jit
  def f(x):
    b = x + x
    print(b)
    return b.get_field(3.)

  z = f(a)
  print(z)
  print(make_jaxpr(f)(a))
  
  
  @jit
  def f(x):
    b = x + x
    print(b)
    return b.get_field_on_grid()
  
  print(make_jaxpr(f)(x))
  
if __name__ == '__main__':
  with jax.checking_leaks():
    test_continous_gradient()
    test_jit_continous_gradient()