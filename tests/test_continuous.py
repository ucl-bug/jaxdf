import numpy as np
import pytest
from jax import jit
from jax import numpy as jnp
from jax import random

from jaxdf.discretization import Continuous
from jaxdf.geometry import Domain

PRNGKEY = random.PRNGKey(42)


def _test_binary_operator(operator, a, b, jitting, test_point, expected_value):
  operator = jit(operator) if jitting else operator
  z = operator(a, b)
  if jitting:
    z = operator(a, b)
  pred_value = z(test_point)
  # print('--------', pred_value)
  # print('--------', expected_value)
  assert np.allclose(pred_value, expected_value)


@pytest.mark.parametrize("N", [(64, 64)])
@pytest.mark.parametrize("out_dims", [1, 3])
@pytest.mark.parametrize("jitting", [True, False])
def test_create_field(N, out_dims, jitting):
  domain = Domain(N, dx=[1.0] * len(N))

  def get_fun(params, x):
    return jnp.dot(params, x)

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


@pytest.fixture(params=[(64, 64)])
def N(request):
  return request.param


@pytest.fixture
def get_fields(N):
  domain = Domain(N, dx=[1.0] * len(N))

  def f(p, x):
    return jnp.expand_dims(jnp.sum(p * (x**2)), -1)

  # Defining fields
  a = Continuous(5.0, domain, f)
  b = Continuous(6.0, domain, f)
  return a, b


@pytest.mark.parametrize("jitting", [True, False])
def test_add(N, jitting, get_fields):
  a, b = get_fields

  # Testing add of fields
  _test_binary_operator(
      lambda x, y: x + y,
      a,
      b,
      jitting,
      a.domain.origin + 1,
      expected_value=[11.0 * len(N)],
  )

  # Testing add with scalar
  _test_binary_operator(
      lambda x, y: x + 1.0,
      a,
      b,
      jitting,
      a.domain.origin + 1,
      expected_value=[5.0 * len(N) + 1],
  )


@pytest.mark.parametrize("jitting", [True, False])
def test_equality(jitting):
  get_fun = lambda params, x: jnp.dot(params, x)
  second_get_fun = lambda params, x: jnp.dot(params, x) + 1.0

  a = Continuous(5.0, Domain((64, 64), dx=(1.0, 1.0)), get_fun)
  b = Continuous(5.0, Domain((64, 64), dx=(1.0, 1.0)), get_fun)
  c = Continuous(6.0, Domain((64, 64), dx=(1.0, 1.0)), get_fun)
  d = Continuous(5.0, Domain((64, 64), dx=(1.0, 1.0)), second_get_fun)
  e = Continuous(5.0, Domain((64, 64), dx=(2.0, 1.0)), get_fun)

  def check_equal(a, b):
    return a == b

  if jitting:
    check_equal = jit(check_equal)

  assert check_equal(a, b) == True
  assert check_equal(a, c) == False
  assert check_equal(a, d) == False
  assert check_equal(a, e) == False


def test_from_function():

  def f(params, x):
    return jnp.sum(params) * x

  def init_params(rng, domain):
    return jnp.ones((1, ))

  domain = Domain((64, ), dx=(1.0, ))
  seed = random.PRNGKey(42)
  field = Continuous.from_function(domain, init_params, f, seed)


def test_on_grid():

  def f(params, x):
    return params * jnp.sum(x)

  params = jnp.ones((1, ))

  domain = Domain((4, ), dx=(0.1, ))
  field = Continuous(params, domain, f)
  grid = field.on_grid
  grid_true = jnp.asarray([[-0.15], [-0.05], [0.05], [0.15]])
  assert jnp.allclose(grid, grid_true)


def test_is_complex():

  def f(params, x):
    return params * jnp.sum(x)

  def g(params, x):
    return 1j * params * jnp.sum(x)

  params = jnp.ones((1, ))

  domain = Domain((4, ), dx=(0.1, ))
  real_field = Continuous(params, domain, f)
  complex_field = Continuous(params, domain, g)

  assert real_field.is_complex == False
  assert complex_field.is_complex == True


def test_replace_params():

  def f(params, x):
    return params * jnp.sum(x)

  params = jnp.ones((1, ))

  domain = Domain((4, ), dx=(0.1, ))
  field = Continuous(params, domain, f)
  field = field.replace_params(jnp.zeros((1, )))
  assert jnp.allclose(field.params, jnp.zeros((1, )))


@pytest.mark.parametrize("jitting", [True, False])
def test_sub(N, jitting, get_fields):
  a, b = get_fields

  # Testing add of fields
  _test_binary_operator(
      lambda x, y: x - y,
      a,
      b,
      jitting,
      a.domain.origin + 1,
      expected_value=[-1.0 * len(N)],
  )

  # Testing add with scalar
  _test_binary_operator(
      lambda x, y: x - 1.0,
      a,
      b,
      jitting,
      a.domain.origin + 1,
      expected_value=[5.0 * len(N) - 1],
  )


@pytest.mark.parametrize("jitting", [True, False])
def test_mul(N, jitting, get_fields):
  a, b = get_fields

  # Testing add of fields
  values = {1: 30.0, 2: 120, 3: 270}
  _test_binary_operator(
      lambda x, y: x * y,
      a,
      b,
      jitting,
      a.domain.origin + 1,
      expected_value=[values[len(N)]],
  )

  # Testing add with scalar
  _test_binary_operator(
      lambda x, y: x * 2.0,
      a,
      b,
      jitting,
      a.domain.origin + 1,
      expected_value=[10.0 * len(N)],
  )


def test_op_neg():

  def f(params, x):
    return params * jnp.sum(x)

  params = jnp.ones((1, ))

  domain = Domain((4, ), dx=(0.1, ))
  field = Continuous(params, domain, f)
  field = -field
  field_value = field(jnp.ones((1, )))

  true_field_value = -f(params, jnp.ones((1, )))
  assert jnp.allclose(field_value, true_field_value)


def test_op_rtruediv():

  def f(params, x):
    return params * jnp.sum(x)

  params = jnp.ones((1, )) * 4.0

  domain = Domain((4, ), dx=(0.1, ))
  field = Continuous(params, domain, f)
  field = 1.0 / field
  field_value = field(jnp.ones((1, )))

  true_field_value = 1.0 / f(params, jnp.ones((1, )))
  assert jnp.allclose(field_value, true_field_value)


def test_op_truediv():

  def f(params, x):
    return params * jnp.sum(x)

  def g(params, x):
    return params * jnp.sum(x) + 2.0

  domain = Domain((4, ), dx=(0.1, ))
  field1 = Continuous(4.0, domain, f)
  field2 = Continuous(2.0, domain, g)
  field = field1 / field2
  field_value = field(jnp.ones((1, )))
  true_field_value = f(4.0, jnp.ones((1, ))) / g(2.0, jnp.ones((1, )))
  assert jnp.allclose(field_value, true_field_value)


def test_op_truediv_float():

  def f(params, x):
    return params * jnp.sum(x)

  domain = Domain((4, ), dx=(0.1, ))
  field = Continuous(4.0, domain, f)
  field = field / 2.0
  field_value = field(jnp.ones((1, )))
  true_field_value = f(4.0, jnp.ones((1, ))) / 2.0
  assert jnp.allclose(field_value, true_field_value)


def test_op_pow():

  def f(params, x):
    return params * jnp.sum(x)

  def g(params, x):
    return params * jnp.sum(x) + 2.0

  domain = Domain((4, ), dx=(0.1, ))
  field1 = Continuous(4.0, domain, f)
  field2 = Continuous(2.0, domain, g)
  field = field1**field2
  field_value = field(jnp.ones((1, )))
  true_field_value = f(4.0, jnp.ones((1, )))**g(2.0, jnp.ones((1, )))
  assert jnp.allclose(field_value, true_field_value)


def test_op_pow_float():

  def f(params, x):
    return params * jnp.sum(x)

  domain = Domain((4, ), dx=(0.1, ))
  field = Continuous(4.0, domain, f)
  field = field**2.0
  field_value = field(jnp.ones((1, )))
  true_field_value = f(4.0, jnp.ones((1, )))**2.0
  assert jnp.allclose(field_value, true_field_value)
