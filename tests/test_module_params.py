"""Tests that eqx.Module objects work as operator params with jit/grad/vmap."""
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxdf import OnGrid, operator
from jaxdf.geometry import Domain


class SimpleModule(eqx.Module):
  weight: jax.Array
  bias: jax.Array

  def __call__(self, x):
    return self.weight * x + self.bias


def module_init(x: OnGrid, *args, **kwargs):
  return SimpleModule(weight=jnp.array(2.0), bias=jnp.array(1.0))


@operator(init_params=module_init)
def module_op(x: OnGrid, *, params=None):
  """Operator that uses an eqx.Module as params."""
  return x.replace_params(params(x.params))


@pytest.fixture
def field():
  domain = Domain((8, ), (1.0, ))
  return OnGrid(jnp.ones((8, 1)) * 3.0, domain)


def test_module_params_basic(field):
  """eqx.Module as operator params works."""
  result = module_op(field)
  expected = 2.0 * 3.0 + 1.0    # weight * value + bias
  assert jnp.allclose(result.params, expected)


def test_module_params_explicit(field):
  """Explicitly passing eqx.Module as params works."""
  custom = SimpleModule(weight=jnp.array(5.0), bias=jnp.array(0.0))
  result = module_op(field, params=custom)
  assert jnp.allclose(result.params, 15.0)


def test_module_params_default_params(field):
  """default_params returns the eqx.Module."""
  params = module_op.default_params(field)
  assert isinstance(params, SimpleModule)
  assert params.weight == 2.0


def test_module_params_jit(field):
  """eqx.Module params work with jax.jit."""

  @jax.jit
  def f(x):
    return module_op(x)

  result = f(field)
  expected = 2.0 * 3.0 + 1.0
  assert jnp.allclose(result.params, expected)


def test_module_params_jit_explicit(field):
  """Explicitly passed eqx.Module params work with jax.jit."""
  custom = SimpleModule(weight=jnp.array(5.0), bias=jnp.array(0.0))

  @jax.jit
  def f(x, params):
    return module_op(x, params=params)

  result = f(field, custom)
  assert jnp.allclose(result.params, 15.0)


def test_module_params_grad(field):
  """jax.grad flows through eqx.Module operator params."""

  def loss(module_params):
    result = module_op(field, params=module_params)
    return jnp.sum(result.params)

  custom = SimpleModule(weight=jnp.array(3.0), bias=jnp.array(0.0))
  grads = jax.grad(loss)(custom)

  # d/d(weight) of sum(weight * x + bias) = sum(x)
  assert isinstance(grads, SimpleModule)
  assert jnp.allclose(grads.weight, jnp.sum(field.params))
  # d/d(bias) = n_elements
  assert jnp.allclose(grads.bias, float(field.params.size))


def test_module_params_vmap(field):
  """jax.vmap works with eqx.Module operator params (batched over params)."""
  weights = jnp.array([1.0, 2.0, 3.0])
  biases = jnp.array([0.0, 0.0, 0.0])
  batched_modules = jax.vmap(SimpleModule)(weights, biases)

  def apply_one(params):
    return module_op(field, params=params).params

  results = jax.vmap(apply_one)(batched_modules)
  assert results.shape == (3, 8, 1)
  assert jnp.allclose(results[0], 3.0)    # 1.0 * 3.0
  assert jnp.allclose(results[1], 6.0)    # 2.0 * 3.0
  assert jnp.allclose(results[2], 9.0)    # 3.0 * 3.0


def test_module_params_jit_grad(field):
  """jit + grad combined works with eqx.Module params."""

  @jax.jit
  def loss(module_params):
    result = module_op(field, params=module_params)
    return jnp.sum(result.params)

  custom = SimpleModule(weight=jnp.array(3.0), bias=jnp.array(0.0))
  grads = jax.grad(loss)(custom)

  assert isinstance(grads, SimpleModule)
  assert jnp.isfinite(grads.weight)
  assert jnp.isfinite(grads.bias)
