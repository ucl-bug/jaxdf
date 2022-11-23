import jax
from jax import jit, make_jaxpr
from jax import numpy as jnp

from jaxdf import *

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

  op_params = operators.gradient.default_params(m)
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

  op_params = operators.laplacian.default_params(m)
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

  op_params = operators.gradient.default_params(a)
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

  op_params = operators.laplacian.default_params(a)
  _ = (f(a, op_params).params)


def test_jit_continous_gradient():
  @jit
  def f(m, op_params):
    return operators.gradient(m, params=op_params)

  op_params = operators.gradient.default_params(m)
  _ = (f(m, op_params).params[...,0])

  op_params_a = operators.gradient.default_params(a)
  _ = (f(a, op_params_a).params)
  _ = (make_jaxpr(f)(a, op_params_a))
  _ = (make_jaxpr(f)(m, op_params))

def test_continuous_derivative():
  def f(params, x):
    return jnp.sin(params*x)
  domain = geometry.Domain((8,), (1.0,))
  a = Continuous(1.5, domain, f)
  b = operators.derivative(a)
  da = lambda p, x : p*jnp.cos(p*x)

  p = jnp.asarray([2.0])

  f1 = b(p)
  f2 = da(1.5, p)
  assert jnp.allclose(f1, f2)

def test_finite_difference_derivative():
  domain = geometry.Domain((11,), (.5,))
  params = jnp.zeros((11,))
  params = params.at[5].set(1.0)
  u = FiniteDifferences(params, domain, accuracy=4)
  du = operators.gradient(u)

  kernel = operators.gradient.default_params(u)
  print(kernel)

  grid_values = du.on_grid[:,0]
  true_values = jnp.asarray(
    [ 0., -0.,  0. , -0.16666667,  1.33333333, 0.,
      -1.33333333, 0.16666667,  0.,  0., 0.])
  assert jnp.allclose(grid_values, true_values)

def test_checking_leaks():
  with jax.checking_leaks():
    test_fourier_laplacian()
    test_continuous_laplacian()
    test_fourier_gradient()
    test_continous_gradient()
    test_jit_continous_gradient()#

if __name__ == "__main__":
  test_finite_difference_derivative()
