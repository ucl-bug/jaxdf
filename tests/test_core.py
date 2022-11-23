import jax
import numpy as np
import pytest
from jax import jit
from jax import numpy as jnp

from jaxdf import *


@pytest.fixture
def get_ongrid_fields():
  domain = geometry.Domain((1,), (1.,))
  a = OnGrid(jnp.asarray([1.0]), domain)
  b = OnGrid(jnp.asarray([2.0]), domain)
  return a, b

@pytest.fixture
def get_continuous_fields():
  domain = geometry.Domain()
  # Continuous fields
  def f(p, x):
    return jnp.expand_dims(jnp.sum(p*(x**2)), -1)
  a = Continuous(5.0, domain, f)
  b = Continuous(6.0, domain, f)
  return a, b

def test_jit_call(get_continuous_fields):
  a, b = get_continuous_fields

  @jit
  def f(x):
    q = x + 2
    return q.get_field(x.domain.origin)
  z = f(a)
  z = f(a)
  assert np.allclose(z, [2.])

def test_call_field(get_continuous_fields):
  a, b = get_continuous_fields

  z = a(a.domain.origin)

  @jit
  def f(u):
    return u(u.domain.origin)

  z = f(a)
  z = f(a)

def test_make_fourier_inside_jitted_fun(get_ongrid_fields):
  a, b = get_ongrid_fields
  @jax.jit
  def f(x):
    y = FourierSeries(x.params**2, x.domain)
    return y + 1

  v = f(a)
  v = f(a)
  assert v.params == 2.

def test_override_operator_new_discretization(get_ongrid_fields):
  a, b = get_ongrid_fields

  z = operators.compose(a)(jnp.exp)
  z_old = z.params


  class MyDiscr(OnGrid):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)


  @operator
  def compose(x: MyDiscr, *, params=None):
    def decorator(fun):
      return x.replace_params(fun(x.params) + 10)
    return decorator, None

  a_new = MyDiscr(a.params, a.domain)

  z = operators.compose(a_new)(jnp.exp)
  print(z.params, jnp.exp(1.0) + 10)
  assert z.params == jnp.exp(1.0) + 10

  z = operators.compose(a)(jnp.exp)
  assert z.params ==  z_old

def test_replace_param_for_abstract_field():
  domain = geometry.Domain((1,), (1.,))
  params = jnp.asarray([1.0])
  aux = {"a": 1, "f": lambda x: x, "s": "string"}
  dims = 1

  a = Field(params, domain, dims, aux)

  new_params = jnp.asarray([2.0])

  b = a.replace_params(new_params)

  assert b.params == new_params
  assert b.aux == aux
  assert b.dims == dims
  assert b.domain == domain

@pytest.mark.parametrize("function", [
  "__add__",
  "__radd__",
  "__sub__",
  "__rsub__",
  "__mul__",
  "__rmul__",
  "__pow__",
  "__rpow__",
  "__truediv__",
  "__rtruediv__",
])
def test_non_implemented_binary_methods(function):
  domain = geometry.Domain((1,), (1.,))
  params = jnp.asarray([1.0])
  aux = {"a": 1, "f": lambda x: x, "s": "string"}
  dims = 1

  a = Field(params, domain, dims, aux)
  b = Field(params, domain, dims, aux)
  c = FourierSeries(params, domain)

  with pytest.raises(NotImplementedError):
    getattr(a, function)(b)

  with pytest.raises(NotImplementedError):
    getattr(a, function)(c)

  with pytest.raises(NotImplementedError):
    getattr(c, function)(a)

@pytest.mark.parametrize("function", [
  "__neg__",
])
def test_non_implemented_unary_methods(function):
  domain = geometry.Domain((1,), (1.,))
  params = jnp.asarray([1.0])
  aux = {"a": 1, "f": lambda x: x, "s": "string"}
  dims = 1

  a = Field(params, domain, dims, aux)

  with pytest.raises(NotImplementedError):
    getattr(a, function)()
