"""
Unit tests for Helmholtz equation functionality.

Tests extracted from example_1_paper.ipynb and simulate_helmholtz_equation.ipynb
to provide fast unit test coverage without running full optimization loops.
"""
import jax
import jax.numpy as jnp
import pytest
from jax.scipy.sparse.linalg import gmres

from jaxdf.discretization import Continuous, FourierSeries
from jaxdf.geometry import Domain
from jaxdf.operators import (
    compose,
    diag_jacobian,
    gradient,
    laplacian,
    sum_over_dims,
)


# Helper functions (DRY principle - define once, use everywhere)
def pml_sigma(x, alpha=2.0, sigma_star=1.0, delta_pml=12.0, L_half=16.0):
  """PML absorption function."""
  abs_x = jnp.abs(x)
  in_pml_amplitude = (jnp.abs(abs_x - delta_pml) / (L_half - delta_pml))**alpha
  return jnp.where(abs_x > delta_pml, sigma_star * in_pml_amplitude, 0.0)


def pml_gamma(x, omega=1.0, **kwargs):
  """PML gamma function: 1/(1 + i*sigma/omega)."""
  y = compose(x)(lambda x_val: pml_sigma(x_val, **kwargs))
  return 1.0 / (1.0 + 1j * y / omega)


def modified_laplacian(u, pml):
  """Modified Laplacian with PML: ∇·(pml ∇(pml u))."""
  grad_u = gradient(u, correct_nyquist=False)
  mod_grad_u = grad_u * pml
  mod_diag_jacobian = diag_jacobian(mod_grad_u, correct_nyquist=False) * pml
  return sum_over_dims(mod_diag_jacobian)


def helmholtz_operator(u, c, pml, omega=1.0):
  """Helmholtz operator: modified_laplacian + k²u."""
  L = modified_laplacian(u, pml)
  k = ((omega / c)**2) * u
  return L + k


def total_variation(u):
  """Total variation: mean(|∇²u|)."""
  nabla_u = laplacian(u)
  absval = compose(nabla_u)(jnp.abs)
  return jnp.mean(absval.on_grid)


# Fixtures
@pytest.fixture
def small_domain():
  """32x32 domain for fast tests."""
  return Domain((32, 32), (1.0, 1.0))


@pytest.fixture
def medium_domain():
  """64x64 domain for tests needing more PML space."""
  return Domain((64, 64), (1.0, 1.0))


# Tests
def test_pml_absorption_sigma(medium_domain):
  """Test PML absorption function sigma(x) works correctly."""
  x = Continuous(None, medium_domain, lambda p, x: x)
  sigma_field = compose(x)(
      lambda x_val: pml_sigma(x_val, delta_pml=25.0, L_half=32.0))

  sigma_on_grid = sigma_field.on_grid

  # Verify sigma is 0 in the center, nonzero near edges
  center_val = sigma_on_grid[32, 32, 0]
  edge_val = sigma_on_grid[0, 0, 0]

  assert jnp.abs(center_val) < 1e-6, "Sigma should be ~0 in center"
  assert edge_val > 0, "Sigma should be > 0 near edges (in PML)"


def test_pml_gamma_function(medium_domain):
  """Test gamma = 1/(1 + i*sigma/omega) with compose operator."""
  x = Continuous(None, medium_domain, lambda p, x: x)
  pml = pml_gamma(x, omega=1.0, delta_pml=25.0, L_half=32.0)

  pml_on_grid = pml.on_grid
  assert jnp.iscomplexobj(pml_on_grid)

  # In center (no PML), gamma should be ~1
  center_val = pml_on_grid[32, 32, 0]
  assert jnp.abs(center_val - 1.0) < 1e-5


def test_modified_laplacian_with_pml(small_domain):
  """Test modified Laplacian: ∇·(pml ∇(pml u))."""
  x = Continuous(None, small_domain, lambda p, x: x)
  pml = pml_gamma(x)
  pml_on_grid = pml.on_grid

  # Create a test field
  params = jnp.zeros((32, 32, 1)) + 0j
  params = params.at[16, 8].set(1.0)

  u = FourierSeries(params, small_domain)
  pml_fs = FourierSeries(pml_on_grid, small_domain)

  # Apply modified Laplacian
  result = modified_laplacian(u, pml_fs)

  assert result is not None
  assert hasattr(result, 'on_grid')
  assert result.on_grid.shape == u.on_grid.shape


def test_helmholtz_operator_basic(small_domain):
  """Test Helmholtz: modified_laplacian + k²u."""
  x = Continuous(None, small_domain, lambda p, x: x)
  pml = pml_gamma(x)
  pml_on_grid = pml.on_grid

  params = jnp.zeros((32, 32, 1)) + 0j
  params = params.at[16, 8].set(1.0)

  u = FourierSeries(params, small_domain)
  pml_fs = FourierSeries(pml_on_grid, small_domain)
  c = FourierSeries(params + 1.0, small_domain)

  result = helmholtz_operator(u, c, pml_fs, omega=0.5)

  assert result is not None
  assert hasattr(result, 'on_grid')
  assert result.on_grid.shape == u.on_grid.shape


def test_helmholtz_with_fourier_series(small_domain):
  """Test Helmholtz with FourierSeries discretization."""
  x = Continuous(None, small_domain, lambda p, x: x)
  pml = pml_gamma(x)
  pml_on_grid = pml.on_grid

  params = jnp.zeros((32, 32, 1)) + 0j
  params = params.at[16, 8].set(1.0)

  u = FourierSeries(params, small_domain)
  pml_fs = FourierSeries(pml_on_grid, small_domain)
  c = FourierSeries(params + 1.5, small_domain)

  result = helmholtz_operator(u, c, pml_fs)

  assert isinstance(result, FourierSeries)
  assert result.on_grid.shape == u.on_grid.shape


def test_gmres_solver_integration(small_domain):
  """Test GMRES solver works with Helmholtz operator."""
  x = Continuous(None, small_domain, lambda p, x: x)
  pml = pml_gamma(x)
  pml_on_grid = pml.on_grid

  params = jnp.zeros((32, 32, 1)) + 0j
  src_params = params.at[16, 8].set(1.0)

  pml_fs = FourierSeries(pml_on_grid, small_domain)
  c = FourierSeries(params + 1.5, small_domain)
  src = FourierSeries(src_params, small_domain)

  # Wrapper for GMRES
  def helm_func(u):
    return helmholtz_operator(u, c, pml_fs)

  # Solve with GMRES (just a few iterations to test integration)
  sol, info = gmres(helm_func, src, maxiter=10, tol=1e-3)

  assert sol is not None
  assert hasattr(sol, 'on_grid')
  assert sol.on_grid.shape == src.on_grid.shape


def test_total_variation_operator(small_domain):
  """Test total variation computation on fields."""
  params = jnp.zeros((32, 32, 1))
  params = params.at[10:20, 10:20].set(1.0)
  u = FourierSeries(params, small_domain)

  tv = total_variation(u)

  assert jnp.ndim(tv) == 0
  assert tv >= 0    # TV should be non-negative


def test_helmholtz_complex_valued(small_domain):
  """Test Helmholtz with complex-valued fields."""
  x = Continuous(None, small_domain, lambda p, x: x)
  pml = pml_gamma(x)
  pml_on_grid = pml.on_grid

  params = jnp.zeros((32, 32, 1), dtype=jnp.complex64)
  params = params.at[16, 8].set(1.0 + 0.5j)

  u = FourierSeries(params, small_domain)
  pml_fs = FourierSeries(pml_on_grid, small_domain)
  c = FourierSeries(jnp.zeros((32, 32, 1)) + 1.5, small_domain)

  result = helmholtz_operator(u, c, pml_fs)

  assert jnp.iscomplexobj(result.on_grid)
  assert result.on_grid.shape == u.on_grid.shape


def test_helmholtz_jit_compilation(small_domain):
  """Test Helmholtz operator can be jit compiled."""
  x = Continuous(None, small_domain, lambda p, x: x)
  pml = pml_gamma(x)
  pml_on_grid = pml.on_grid

  params = jnp.zeros((32, 32, 1)) + 0j
  params = params.at[16, 8].set(1.0)

  u = FourierSeries(params, small_domain)
  pml_fs = FourierSeries(pml_on_grid, small_domain)
  c = FourierSeries(params + 1.5, small_domain)

  # JIT compile the Helmholtz operator
  jitted_helmholtz = jax.jit(helmholtz_operator)
  result = jitted_helmholtz(u, c, pml_fs)

  assert result is not None
  assert result.on_grid.shape == u.on_grid.shape


def test_helmholtz_gradient_computation(small_domain):
  """Test jax.grad works with Helmholtz loss function."""

  def loss(c_params):
    c = FourierSeries(c_params + 1.0, small_domain)
    tv = total_variation(c)
    return jnp.real(tv)

  test_params = jnp.zeros((32, 32, 1))
  test_params = test_params.at[10:20, 10:20].set(0.5)

  grad_fn = jax.grad(loss)
  grads = grad_fn(test_params)

  assert grads.shape == test_params.shape
