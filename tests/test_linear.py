import pytest
from jax import numpy as jnp

from jaxdf.discretization import FiniteDifferences, Linear
from jaxdf.geometry import Domain


@pytest.mark.parametrize("N", [(33, ), (33, 33), (33, 33, 33)])
def test_create(N):
  domain = Domain(N, dx=[1.0] * len(N))
  params = jnp.zeros(N)
  field = Linear(params, domain)
  assert field.params.shape == N


def test_op_neg():
  domain = Domain((1, ), (1.0, ))
  a = Linear(jnp.asarray([1.0]), domain)

  b = (-a).params
  b_exp = -(a.params)
  assert b == b_exp


def test_equality():
  domain = Domain((1, ), (1.0, ))
  a = Linear(jnp.asarray([1.0]), domain)
  b = Linear(jnp.asarray([1.0]), domain)
  assert a == b

  c = Linear(jnp.asarray([2.0]), domain)
  assert a != c

  d = Linear(jnp.asarray([1.0]), Domain((2, ), (1.0, )))
  assert a != d


def test_equality_with_type():
  """Regression test for jaxdf#145 — __eq__ should not crash when compared with a type."""
  domain = Domain((1, ), (1.0, ))
  a = Linear(jnp.asarray([1.0]), domain)
  assert (a == Linear) == False
  assert (a == int) == False


@pytest.mark.parametrize("accuracy", [2, 4, 8, 16])
def test_fd_replace_params_preserves_accuracy(accuracy):
  """Regression test for jwave#224 — replace_params must preserve accuracy."""
  domain = Domain((10, ), (1.0, ))
  field = FiniteDifferences(jnp.zeros(10), domain, accuracy=accuracy)
  new_field = field.replace_params(jnp.ones(10))
  assert new_field.accuracy == accuracy
  assert jnp.allclose(new_field.params, jnp.ones(10))


def test_fd_replace_params_in_scan():
  """Regression test for jwave#224 — lax.scan must not fail when replace_params
  is used on FiniteDifferences with non-default accuracy, since the carry
  pytree metadata (accuracy) must stay consistent between iterations."""
  import jax

  domain = Domain((10, ), (1.0, ))
  field = FiniteDifferences(jnp.zeros(10), domain, accuracy=4)

  def scan_fn(carry, _):
    new_carry = carry.replace_params(carry.params + 1.0)
    return new_carry, None

  result, _ = jax.lax.scan(scan_fn, field, jnp.arange(3))
  assert result.accuracy == 4
  assert jnp.allclose(result.params, jnp.full(10, 3.0))
