import jax.numpy as jnp
import pytest
from jax import jit

from jaxdf.discretization import FourierSeries
from jaxdf.geometry import Domain


@pytest.mark.parametrize("N", [(33, ), (33, 33), (33, 33, 33)])
@pytest.mark.parametrize("jitting", [True, False])
@pytest.mark.parametrize("out_dims", [1, 3])
def test_call(N, out_dims, jitting):
  domain = Domain(N, dx=[1.0] * len(N))
  true_size = list(N) + [out_dims]
  params = jnp.zeros(true_size)

  delta_position = [x // 2 for x in N]
  if len(N) == 1:
    params = params.at[delta_position[0], :].set(1.0)
  elif len(N) == 2:
    params = params.at[delta_position[0], delta_position[1], :].set(1.0)
  elif len(N) == 3:
    params = params.at[delta_position[0], delta_position[1],
                       delta_position[2], :].set(1.0)

  value = jnp.asarray([1.0] * out_dims)
  x = jnp.asarray([0.0] * len(N))

  def get(params, x):
    field = FourierSeries(params, domain)
    return field(x)

  get = jit(get) if jitting else get

  field_value = get(params, x)
  if jitting:
    field_value = get(params, x)

  assert jnp.allclose(field_value, value)


def test_ffts_funcs():
  domain = Domain((33, ), dx=[1.0])
  params = jnp.zeros((33, 1))
  field = FourierSeries(params, domain)
  ffts = field._ffts
  assert ffts == [jnp.fft.rfft, jnp.fft.irfft]

  params = params + 1j
  field = FourierSeries(params, domain)
  ffts = field._ffts
  assert ffts == [jnp.fft.fft, jnp.fft.ifft]


def test_cut_freq_axis():
  domain = Domain((6, 6), dx=[1.0])
  params = jnp.zeros((6, 6))
  field = FourierSeries(params, domain)

  cut_freq = field._cut_freq_axis[0]
  ref = jnp.asarray([0., 1.0471976, 2.0943952, 3.1415927])
  assert jnp.allclose(cut_freq, ref)
