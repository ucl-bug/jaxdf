from re import A
from jaxdf import *
from jax import jit, make_jaxpr, grad
from jax import numpy as jnp
import inspect
import jax


domain = geometry.Domain((8,8), (.5,.5))

# Fields on grid
x = OnGrid(jnp.asarray([1.0]), domain)
y = OnGrid(jnp.asarray([2.0]), domain)

def f(p, x):
  return jnp.expand_dims(jnp.sum(p*(x**2)), -1)
a = Continuous(3.0, domain, f)
b = Continuous(6.0, domain, f)

# Dirc delta FourierSeries field
m = FourierSeries.empty(domain)
p = m.params.at[4,4].set(1.0)
m = FourierSeries(p, domain)

def test_fourier_gradient():
  _ = (m.params[...,0])
  z = operators.gradient(m)
  _ = (z.params[...,0])
  _ = (z.params[...,1])

  @jit
  def f(m):
    return operators.gradient(m)
  
  _ = (f(m).params[...,0])
  _ = (f(m).params[...,1])
  
  _ = (f(a))
  _ = (make_jaxpr(f)(a))
  _ = (make_jaxpr(f)(m))
  
  op_params = operators.gradient(m)._op_params
  _ = ('op_params', op_params)
  z = operators.gradient(m, params=op_params)
  _ = (z.params[...,0])
  
def test_fourier_laplacian():
  _ = (m.params[...,0])
  z = operators.laplacian(m)
  _ = (z.params[...,0])

  @jit
  def f(m):
    return operators.laplacian(m)
  
  _ = (f(m).params[...,0])
  
  op_params = operators.laplacian(m)._op_params
  _ = ('op_params', op_params)
  z = operators.laplacian(m, params=op_params)
  _ = (z.params[...,0])
  
  @jit
  def f(m, op_params):
    return operators.laplacian(m, params=op_params)
  _ = (f(m, op_params).params[...,0])
  
  
def test_continous_gradient():
  x = jnp.asarray([1.])
  z = operators.gradient(a)
  _ = (z)
  _ = (a.get_field(x), z.get_field(x))
  
  @jit
  def f(a, op_params):
    return operators.gradient(a, params=op_params)
  
  op_params = operators.gradient(a)._op_params
  _ = (f(a, op_params).params)
  

def test_continuous_laplacian():
  x = jnp.asarray([1.])
  z = operators.laplacian(a)
  _ = (z)
  _ = (a.get_field(x), z.get_field(x))
  print(_)
  
  @jit
  def f(a, op_params):
    return operators.laplacian(a, params=op_params)
  
  op_params = operators.laplacian(a)._op_params
  _ = (f(a, op_params).params)
  
  
def test_jit_continous_gradient():
  @jit
  def f(m, op_params):
    return operators.gradient(m, params=op_params)
  
  op_params = operators.gradient(m)._op_params
  _ = (f(m, op_params).params[...,0])

  op_params_a = operators.gradient(a)._op_params
  _ = (f(a, op_params_a).params)
  _ = (make_jaxpr(f)(a, op_params_a))
  _ = (make_jaxpr(f)(m, op_params))
  
if __name__ == '__main__':
  with jax.checking_leaks():
    test_fourier_laplacian()
    test_continuous_laplacian()
    test_fourier_gradient()
    test_continous_gradient()
    test_jit_continous_gradient()