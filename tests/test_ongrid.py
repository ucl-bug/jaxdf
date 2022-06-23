
import pytest
from jax import jit
from jax import numpy as jnp
from jax import random

from jaxdf.discretization import FiniteDifferences, FourierSeries, OnGrid
from jaxdf.geometry import Domain

PRNGKEY = random.PRNGKey(42)

# Tests the initialization of fields OnGrid (and subclasses)
@pytest.mark.parametrize("N", [(64,), (64,64), (64, 64, 64)])
@pytest.mark.parametrize("discretization", [
  OnGrid, FourierSeries, FiniteDifferences
])
@pytest.mark.parametrize("out_dims", [0, 1, 3])
@pytest.mark.parametrize("jitting", [True, False])
def test_create_field(
  N, discretization, out_dims, jitting
):
  domain = Domain(N, dx=[1.]*len(N))

  true_size = list(N)
  if out_dims == 0:
    true_size += [1]
  else:
    true_size += [out_dims]

  def get(key):
    params = jnp.ones(domain.N)
    if out_dims > 0:
      params = jnp.expand_dims(params, -1)
    if out_dims > 1:
      params = jnp.concatenate([params]*out_dims, -1)

    field = discretization(params, domain)
    return field

  get = jit(get) if jitting else get

  # Twice for getting leaked tracers if jitting
  field = get(PRNGKEY)
  if jitting:
    field = get(PRNGKEY)

  assert field.params.shape == tuple(true_size)

@pytest.mark.parametrize("discretization", [
  OnGrid, FourierSeries, FiniteDifferences
])
def test_add(discretization):
  N = (1,)
  domain = Domain(N, dx=[1.]*len(N))
  x = discretization(jnp.asarray([1.0]), domain)
  y = discretization(jnp.asarray([2.0]), domain)

  z = x + y
  assert z.params == 3.0
  assert type(z) == discretization

  @jit
  def fun(x, y):
    return x + y

  z = fun(x, y)
  z = fun(x, y)
  assert z.params == 3.0
  assert type(z) == discretization

  z = x + 5.
  assert z.params == 6.0

  @jit
  def fun(x):
    return x + 5.

  z = fun(x)
  z = fun(x)
  assert z.params == 6.0

@pytest.mark.parametrize("discretization", [
  OnGrid, FourierSeries, FiniteDifferences
])
def test_sub(discretization):
  N = (1,)
  domain = Domain(N, dx=[1.]*len(N))
  x = discretization(jnp.asarray([1.0]), domain)
  y = discretization(jnp.asarray([2.0]), domain)

  z = x - y
  assert z.params == -1.0
  assert type(z) == discretization

  @jit
  def fun(x, y):
    return x - y

  z = fun(x, y)
  z = fun(x, y)
  assert z.params == -1.0
  assert type(z) == discretization

  z =  x - 2.0
  assert z.params == -1.0

  @jit
  def fun(x):
    return  x - 2.0

  z = fun(x)
  z = fun(x)
  assert z.params == -1.0

@pytest.mark.parametrize("discretization", [
  OnGrid, FourierSeries, FiniteDifferences
])
def test_jit_with_float(discretization):
  N = (1,)
  domain = Domain(N, dx=[1.]*len(N))
  x = discretization(jnp.asarray([1.0]), domain)
  y = discretization(jnp.asarray([2.0]), domain)

  @jit
  def add(x, y):
    return x + y * 10

  _ = add(x,y)
  _ = add(x, 6.0)
  _ = add(-5.0, x)
