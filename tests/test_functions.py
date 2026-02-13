import jax
import numpy as np
from jax import jit
from jax import numpy as jnp

from jaxdf import *

ATOL = 1e-6

domain = geometry.Domain()

# Fields on grid
x = OnGrid(jnp.asarray([1.0]), domain)
y = OnGrid(jnp.asarray([2.0]), domain)


# Continuous fields
def f(p, x):
  return jnp.expand_dims(jnp.sum(p * (x**2)), -1)


a = Continuous(5.0, domain, f)
b = Continuous(6.0, domain, f)


def test_compose_continuous():
  z = operators.compose(a)(jnp.exp)
  assert np.allclose(z(domain.origin), 1.0)


def test_compose_ongrid():
  x = OnGrid(jnp.asarray([1.0]), domain)
  z = operators.compose(x)(jnp.exp)
  assert z.params == jnp.exp(1.0)


def test_compose_gradient():

  @jit
  def f(x):
    z = operators.compose(x)(jnp.exp)
    print(z.dims)
    return operators.gradient(z)

  print(f(a))
  print(f(a)(domain.origin + 1))


def test_compose():
  x = 2.0
  y = jnp.tanh(x)
  z = operators.compose(x)(jnp.tanh)
  assert np.allclose(z, y)


def test_functional():
  x = 2.0
  y = jnp.sum(x)
  z = operators.functions.functional(x)(jnp.sum)
  assert np.allclose(z, y)


def test_functional_ongrid():
  x = OnGrid(jnp.asarray([1.0]), domain)
  y = jnp.sum(x.on_grid)
  z = operators.functions.functional(x)(jnp.sum)
  assert np.allclose(z, y)


def test_fd_shift_operator():
  domain = geometry.Domain((5, 5), (0.5, 0.5))
  params = jnp.zeros((5, 5, 1))
  params = params.at[2, 2].set(1.0)
  x = FiniteDifferences(params, domain, accuracy=2)

  y = operators.shift_operator(x, dx=[0.125, 0.0]).on_grid[..., 0]
  y_true = jnp.asarray([
      [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
      ],
      [
          0.0,
          0.0,
          0.25,
          0.0,
          0.0,
      ],
      [
          0.0,
          0.0,
          0.75,
          0.0,
          0.0,
      ],
      [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
      ],
      [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
      ],
  ])
  assert np.allclose(y, y_true, atol=ATOL)


def test_fourier_shift_operator():
  domain = geometry.Domain((5, 5), (0.5, 0.5))
  params = jnp.zeros((5, 5, 1))
  params = params.at[2, 2].set(1.0)
  x = FourierSeries(params, domain)

  y = operators.shift_operator(x, dx=[0.125, 0.0]).on_grid[..., 0]
  y_true = jnp.asarray([
      [0.0, 0.0, -0.15872093, 0.0, 0.0],
      [0.0, 0.0, 0.3115073, 0.0, 0.0],
      [0.0, 0.0, 0.9040295, 0.0, 0.0],
      [0.0, 0.0, -0.20000003, 0.0, 0.0],
      [0.0, 0.0, 0.14318419, 0.0, 0.0],
  ])
  assert np.allclose(y, y_true, atol=ATOL)


def test_fd_shift_zero_is_identity():
  """Shifting by dx=0 should return the original field."""
  domain = geometry.Domain((8, ), (0.5, ))
  params = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).reshape(8, 1)
  x = FiniteDifferences(params, domain, accuracy=2)
  y = operators.shift_operator(x, dx=[0.0])
  assert np.allclose(y.on_grid, x.on_grid, atol=ATOL)


def test_fourier_shift_zero_is_identity():
  """Shifting by dx=0 should return the original field."""
  domain = geometry.Domain((8, ), (0.5, ))
  params = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).reshape(8, 1)
  x = FourierSeries(params, domain)
  y = operators.shift_operator(x, dx=[0.0])
  assert np.allclose(y.on_grid, x.on_grid, atol=ATOL)


def test_fd_shift_1d_regression():
  """Pin exact numerical output of 1D FD shift for regression detection."""
  domain = geometry.Domain((8, ), (0.5, ))
  params = jnp.zeros((8, 1))
  params = params.at[4, 0].set(1.0)
  x = FiniteDifferences(params, domain, accuracy=2)

  y = operators.shift_operator(x, dx=[0.25])
  # stagger = 0.25 / 0.5 = 0.5 -> half-grid shift toward lower index
  expected = jnp.array([0., 0., 0., 0.5, 0.5, 0., 0., 0.])
  assert np.allclose(y.on_grid[..., 0], expected, atol=ATOL)


def test_fourier_shift_1d_regression():
  """Pin exact numerical output of 1D Fourier shift for regression detection."""
  domain = geometry.Domain((8, ), (0.5, ))
  params = jnp.zeros((8, 1))
  params = params.at[4, 0].set(1.0)
  x = FourierSeries(params, domain)

  y = operators.shift_operator(x, dx=[0.25])
  expected = jnp.array([
      -0.02486405, 0.0835223, -0.1870757, 0.62841743, 0.6284174, -0.18707569,
      0.08352231, -0.02486402
  ])
  assert np.allclose(y.on_grid[..., 0], expected, atol=1e-5)


def test_fd_shift_per_axis_dx():
  """FD shift uses dx[i] for axis i, not dx[0] for all axes (issue #146)."""
  domain = geometry.Domain((8, 8), (0.5, 0.5))
  params = jnp.zeros((8, 8, 2))
  params = params.at[4, 4, 0].set(1.0)
  params = params.at[4, 4, 1].set(1.0)
  x = FiniteDifferences(params, domain, accuracy=2)

  # Shift only along axis 0, not axis 1
  y = operators.shift_operator(x, dx=[0.25, 0.0])

  # Component 0 is shifted by dx[0]=0.25 along axis 0
  assert np.allclose(y.on_grid[3, 4, 0], 0.5, atol=ATOL)
  assert np.allclose(y.on_grid[4, 4, 0], 0.5, atol=ATOL)

  # Component 1 is shifted by dx[1]=0.0 along axis 1 -> unchanged
  assert np.allclose(y.on_grid[4, 4, 1], 1.0, atol=ATOL)
  assert np.allclose(y.on_grid[4, 3, 1], 0.0, atol=ATOL)


def test_fourier_shift_per_axis_dx():
  """Fourier shift uses dx[i] for axis i (issue #146)."""
  domain = geometry.Domain((8, 8), (0.5, 0.5))
  params = jnp.zeros((8, 8, 2))
  params = params.at[4, 4, 0].set(1.0)
  params = params.at[4, 4, 1].set(1.0)
  x = FourierSeries(params, domain)

  # Shift only along axis 0, not axis 1
  y = operators.shift_operator(x, dx=[0.25, 0.0])

  # Component 1 shifted by 0.0 along axis 1 -> unchanged
  assert np.allclose(y.on_grid[4, 4, 1], 1.0, atol=ATOL)
  assert np.allclose(y.on_grid[4, 3, 1], 0.0, atol=ATOL)

  # Component 0 should have shifted (peak no longer solely at [4,4])
  assert y.on_grid[3, 4, 0] > 0.1    # mass moved toward lower index


def test_fd_shift_broadcast_dx():
  """dx=[a] broadcasts to dx=[a, a] for multi-dim fields (issue #146)."""
  domain = geometry.Domain((8, 8), (0.5, 0.5))
  params = jnp.zeros((8, 8, 2))
  params = params.at[4, 4, 0].set(1.0)
  params = params.at[4, 4, 1].set(1.0)
  x = FiniteDifferences(params, domain, accuracy=2)

  y_broadcast = operators.shift_operator(x, dx=[0.25])
  y_explicit = operators.shift_operator(x, dx=[0.25, 0.25])

  assert np.allclose(y_broadcast.on_grid, y_explicit.on_grid, atol=ATOL)


def test_fourier_shift_broadcast_dx():
  """dx=[a] broadcasts to dx=[a, a] for multi-dim fields (issue #146)."""
  domain = geometry.Domain((8, 8), (0.5, 0.5))
  params = jnp.zeros((8, 8, 2))
  params = params.at[4, 4, 0].set(1.0)
  params = params.at[4, 4, 1].set(1.0)
  x = FourierSeries(params, domain)

  y_broadcast = operators.shift_operator(x, dx=[0.25])
  y_explicit = operators.shift_operator(x, dx=[0.25, 0.25])

  assert np.allclose(y_broadcast.on_grid, y_explicit.on_grid, atol=ATOL)


def test_fd_fourier_shift_direction_consistency():
  """FD and Fourier shift in the same direction for same dx (issue #146)."""
  domain = geometry.Domain((8, 8), (0.5, 0.5))
  params = jnp.zeros((8, 8, 1))
  params = params.at[4, 4, 0].set(1.0)

  x_fd = FiniteDifferences(params, domain, accuracy=2)
  x_fs = FourierSeries(params, domain)

  # Shift along axis 0 only
  y_fd = operators.shift_operator(x_fd, dx=[0.125, 0.0])
  y_fs = operators.shift_operator(x_fs, dx=[0.125, 0.0])

  # Both should move mass toward lower index along axis 0
  fd_col = y_fd.on_grid[:, 4, 0]
  fs_col = y_fs.on_grid[:, 4, 0]

  # Center of mass should be < 4 for both (shifted toward lower index)
  indices = jnp.arange(8, dtype=jnp.float32)
  fd_com = jnp.sum(indices * jnp.abs(fd_col)) / jnp.sum(jnp.abs(fd_col))
  fs_com = jnp.sum(indices * jnp.abs(fs_col)) / jnp.sum(jnp.abs(fs_col))

  assert fd_com < 4.0, f"FD center of mass {fd_com} should be < 4.0"
  assert fs_com < 4.0, f"Fourier center of mass {fs_com} should be < 4.0"


def test_shift_operator_jit_compatible():
  """shift_operator should work inside jit for both discretizations."""
  domain = geometry.Domain((8, ), (0.5, ))
  params = jnp.zeros((8, 1))
  params = params.at[4, 0].set(1.0)

  x_fd = FiniteDifferences(params, domain, accuracy=2)
  x_fs = FourierSeries(params, domain)

  y_fd_jit = jit(lambda x: operators.shift_operator(x, dx=[0.25]))(x_fd)
  y_fs_jit = jit(lambda x: operators.shift_operator(x, dx=[0.25]))(x_fs)

  y_fd = operators.shift_operator(x_fd, dx=[0.25])
  y_fs = operators.shift_operator(x_fs, dx=[0.25])

  assert np.allclose(y_fd_jit.on_grid, y_fd.on_grid, atol=ATOL)
  assert np.allclose(y_fs_jit.on_grid, y_fs.on_grid, atol=ATOL)


def test_jax_leaks():
  with jax.checking_leaks():
    test_compose_continuous()
    test_compose_ongrid()
    test_compose_gradient()


if __name__ == "__main__":
  test_fourier_shift_operator()
