
import pytest
from jax import jit
from jax import numpy as jnp
from jax import random

from jaxdf.discretization import Continuous
from jaxdf.geometry import Domain

PRNGKEY = random.PRNGKey(42)

@pytest.mark.parametrize("N", [(64,), (64,64), (64, 64, 64)])
@pytest.mark.parametrize("out_dims", [1, 3])
@pytest.mark.parametrize("jitting", [True, False])
def test_create_field(N, out_dims, jitting):
  domain = Domain(N, dx=[1.]*len(N))

  def get_fun(params, x):
    return  jnp.dot(params, x)

  keys = random.split(PRNGKEY, 2)
  params = random.uniform(keys[0], (out_dims, len(N)))

  def get(params, key, x):
    field = Continuous(params, domain, get_fun)
    return field(x)

  if jitting:
    get = jit(get)

  field_value = get(params, keys[1], jnp.ones(len(N)))
  if jitting:
    field_value = get(params, keys[1], jnp.ones(len(N)))
