
import numpy as np
import pytest
from jax import jit
from jax import numpy as jnp
from jax import random

from jaxdf.discretization import Continuous
from jaxdf.geometry import Domain

PRNGKEY = random.PRNGKey(42)


def _test_binary_operator(
  operator, a, b, jitting, test_point, expected_value
):
  operator = jit(operator) if jitting else operator
  z = operator(a, b)
  if jitting:
    z = operator(a, b)
  pred_value = z(test_point)
  #print('--------', pred_value)
  #print('--------', expected_value)
  assert np.allclose(pred_value, expected_value)


@pytest.mark.parametrize("N", [(64, 64)])
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


@pytest.fixture(params=[(64,64)])
def N(request):
  return request.param

@pytest.fixture
def get_fields(N):
  domain = Domain(N, dx=[1.]*len(N))

  def f(p, x):
    return jnp.expand_dims(jnp.sum(p*(x**2)), -1)

  # Defining fields
  a = Continuous(5.0, domain, f)
  b = Continuous(6.0, domain, f)
  return a,b

@pytest.mark.parametrize("jitting", [True, False])
def test_add(N, jitting, get_fields):
  a,b = get_fields

  # Testing add of fields
  _test_binary_operator(
    lambda x,y: x+y,
    a, b,
    jitting,
    a.domain.origin+1,
    expected_value=[11.*len(N)]
  )

  # Testing add with scalar
  _test_binary_operator(
    lambda x,y: x+1.,
    a, b,
    jitting,
    a.domain.origin+1,
    expected_value=[5.*len(N) +1]
  )

@pytest.mark.parametrize("jitting", [True, False])
def test_sub(N, jitting, get_fields):
  a,b = get_fields

  # Testing add of fields
  _test_binary_operator(
    lambda x,y: x-y,
    a, b,
    jitting,
    a.domain.origin+1,
    expected_value=[-1.*len(N)]
  )


  # Testing add with scalar
  _test_binary_operator(
    lambda x,y: x-1.,
    a, b,
    jitting,
    a.domain.origin+1,
    expected_value=[5.*len(N) -1]
  )

@pytest.mark.parametrize("jitting", [True, False])
def test_mul(N, jitting, get_fields):
  a,b = get_fields

  # Testing add of fields
  values = {1: 30., 2: 120, 3: 270}
  _test_binary_operator(
    lambda x,y: x*y,
    a, b,
    jitting,
    a.domain.origin+1,
    expected_value=[values[len(N)]])

  # Testing add with scalar
  _test_binary_operator(
    lambda x,y: x*2.,
    a, b,
    jitting,
    a.domain.origin+1,
    expected_value=[10.*len(N)])
