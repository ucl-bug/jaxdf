"""
Unit tests for Burgers equation functionality.

Tests extracted from pinn_burgers.ipynb to provide fast unit test coverage
for Burgers equation patterns with PINNs.
"""
import jax
import jax.numpy as jnp
import pytest
from jax import random
from jax.example_libraries import stax

from jaxdf.discretization import Continuous
from jaxdf.geometry import Domain
from jaxdf.operators.differential import derivative


# Helper functions
def create_burgers_pinn(domain, seed, hidden_dim=20):
  """Create a PINN for Burgers equation."""
  Tanh = stax.elementwise(jnp.tanh)
  init_random_params, predict = stax.serial(
      stax.Dense(hidden_dim),
      Tanh,
      stax.Dense(hidden_dim),
      Tanh,
      stax.Dense(1),
  )

  def init_params(seed, domain):
    return init_random_params(seed, (len(domain.N), ))[1]

  def get_fun(params, x):
    return predict(params, x)

  return Continuous.from_function(domain, init_params, get_fun, seed)


def burgers_operator(u, nu=0.01 / jnp.pi):
  """
    Burgers equation operator:
    du/dt + u*du/dx - ν*d²u/dx² = 0

    Returns the residual (should be 0 when satisfied).
    Assumes domain is [space, time].
    """
  du_dt = derivative(u, axis=1)    # Time derivative
  du_dx = derivative(u, axis=0)    # Space derivative
  ddu_dx = derivative(du_dx, axis=0)    # Second space derivative

  return du_dt + u * du_dx - nu * ddu_dx


# Fixtures
@pytest.fixture
def spacetime_domain():
  """Space-time domain for Burgers equation."""
  return Domain((32, 64), (0.1, 0.01))    # [space, time]


@pytest.fixture
def seed():
  """Random seed for reproducibility."""
  return random.PRNGKey(42)


# Tests
def test_derivative_time_axis(spacetime_domain, seed):
  """Test derivative operator along time axis."""
  u = create_burgers_pinn(spacetime_domain, seed, hidden_dim=10)

  # Compute time derivative (axis=1)
  du_dt = derivative(u, axis=1)

  # Evaluate at a point
  x = jnp.asarray([0.0, 0.0])
  val = du_dt(x)

  assert val is not None
  assert val.shape == (1, )


def test_derivative_space_axis(spacetime_domain, seed):
  """Test derivative operator along space axis."""
  u = create_burgers_pinn(spacetime_domain, seed, hidden_dim=10)

  # Compute space derivative (axis=0)
  du_dx = derivative(u, axis=0)

  # Evaluate at a point
  x = jnp.asarray([0.0, 0.0])
  val = du_dx(x)

  assert val is not None
  assert val.shape == (1, )


def test_second_derivative(spacetime_domain, seed):
  """Test second derivative computation."""
  u = create_burgers_pinn(spacetime_domain, seed, hidden_dim=10)

  # Compute second space derivative
  du_dx = derivative(u, axis=0)
  ddu_dx = derivative(du_dx, axis=0)

  # Evaluate at a point
  x = jnp.asarray([0.0, 0.0])
  val = ddu_dx(x)

  assert val is not None
  assert val.shape == (1, )


def test_burgers_operator_composition(spacetime_domain, seed):
  """Test Burgers equation: du/dt + u*du/dx - ν*d²u/dx²."""
  u = create_burgers_pinn(spacetime_domain, seed, hidden_dim=10)

  # Apply Burgers operator
  residual = burgers_operator(u)

  # Evaluate at a point
  x = jnp.asarray([0.0, 0.0])
  val = residual(x)

  assert val is not None
  assert val.shape == (1, )


def test_burgers_with_continuous_field(spacetime_domain, seed):
  """Test Burgers operator with Continuous discretization."""
  u = create_burgers_pinn(spacetime_domain, seed, hidden_dim=10)

  # Verify field is Continuous
  assert isinstance(u, Continuous)

  # Apply Burgers operator
  residual = burgers_operator(u)

  # Verify residual is also a Continuous field
  assert isinstance(residual, Continuous)


def test_burgers_vmap_residual(spacetime_domain, seed):
  """Test vmap on Burgers residual computation."""
  u = create_burgers_pinn(spacetime_domain, seed, hidden_dim=10)
  domain_sampler = spacetime_domain.domain_sampler
  batch_size = 32

  # Sample points
  coords = domain_sampler(seed, batch_size)

  # Compute residual at all points using vmap
  @jax.vmap
  def compute_residual(x):
    return burgers_operator(u)(x)

  residuals = compute_residual(coords)

  assert residuals.shape == (batch_size, 1)


def test_burgers_boundary_conditions(spacetime_domain, seed):
  """Test boundary condition setup for Burgers equation."""
  u = create_burgers_pinn(spacetime_domain, seed, hidden_dim=10)
  domain_sampler = spacetime_domain.domain_sampler
  batch_size = 64

  def boundary_loss(u, seed, batch_size):
    coords = domain_sampler(seed, batch_size)

    # Initial condition: u(x, t=-T/2) = -sin(pi*x)
    t_min = -0.5 * spacetime_domain.N[1] * spacetime_domain.dx[1]
    coords_t0 = coords.at[:, 1].set(t_min)

    @jax.vmap
    def eval_field(x):
      return u(x)

    field_vals = eval_field(coords_t0)
    target = -jnp.sin(jnp.pi * coords[:, 0:1])
    ic_loss = jnp.mean((field_vals - target)**2)

    # Spatial boundary: u(±L/2, t) = 0
    x_max = 0.5 * spacetime_domain.N[0] * spacetime_domain.dx[0]
    coords_left = coords.at[:, 0].set(-x_max)
    coords_right = coords.at[:, 0].set(x_max)

    left_vals = eval_field(coords_left)
    right_vals = eval_field(coords_right)
    bc_loss = jnp.mean(left_vals**2 + right_vals**2)

    return ic_loss + bc_loss

  loss_val = boundary_loss(u, seed, batch_size)

  assert jnp.ndim(loss_val) == 0
  assert loss_val >= 0


def test_burgers_domain_loss(spacetime_domain, seed):
  """Test domain loss for Burgers equation (PDE residual)."""
  u = create_burgers_pinn(spacetime_domain, seed, hidden_dim=10)
  domain_sampler = spacetime_domain.domain_sampler
  batch_size = 32

  def domain_loss(u, seed, batch_size):
    coords = domain_sampler(seed, batch_size)

    @jax.vmap
    def compute_residual(x):
      return burgers_operator(u)(x)

    residuals = compute_residual(coords)
    return jnp.mean(residuals**2)

  loss_val = domain_loss(u, seed, batch_size)

  assert jnp.ndim(loss_val) == 0
  assert loss_val >= 0


def test_burgers_gradient_computation(spacetime_domain, seed):
  """Test jax.grad works with Burgers PINN loss."""
  domain_sampler = spacetime_domain.domain_sampler
  batch_size = 32

  def total_loss(u, seed):
    coords = domain_sampler(seed, batch_size)

    @jax.vmap
    def compute_residual(x):
      return burgers_operator(u)(x)

    residuals = compute_residual(coords)
    return jnp.mean(residuals**2)

  u = create_burgers_pinn(spacetime_domain, seed, hidden_dim=10)

  # Compute gradient
  grad_fn = jax.grad(total_loss)
  grads = grad_fn(u, seed)

  # Check gradients exist
  assert grads is not None
  assert hasattr(grads, 'params')
