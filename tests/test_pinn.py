"""
Unit tests for Physics-Informed Neural Network (PINN) functionality.

Tests extracted from helmholtz_pinn.ipynb and pinn_burgers.ipynb
to provide fast unit test coverage for PINN patterns.
"""
import jax
import jax.numpy as jnp
import pytest
from jax.example_libraries import stax

from jaxdf.discretization import Continuous
from jaxdf.geometry import Domain
from jaxdf.operators import gradient, laplacian


# Helper functions
def create_simple_nn(input_dim=2, hidden_dim=20, output_dim=1):
  """Create a simple neural network for PINN tests."""
  Tanh = stax.elementwise(jnp.tanh)
  init_random_params, predict = stax.serial(stax.Dense(hidden_dim), Tanh,
                                            stax.Dense(hidden_dim), Tanh,
                                            stax.Dense(output_dim))
  return init_random_params, predict


def create_pinn_field(domain, seed, hidden_dim=20, output_dim=1):
  """Create a Continuous field with neural network function."""
  init_random_params, predict = create_simple_nn(input_dim=len(domain.N),
                                                 hidden_dim=hidden_dim,
                                                 output_dim=output_dim)

  def init_params(seed, domain):
    return init_random_params(seed, (len(domain.N), ))[1]

  def get_fun(params, x):
    return predict(params, x)

  return Continuous.from_function(domain, init_params, get_fun, seed)


# Fixtures
@pytest.fixture
def domain_2d():
  """2D domain for PINN tests."""
  return Domain((32, 32), (1.0, 1.0))


@pytest.fixture
def domain_2d_spacetime():
  """2D space-time domain for Burgers-like tests."""
  return Domain((32, 64), (0.1, 0.01))    # [space, time]


# Tests
def test_continuous_field_with_neural_network(domain_2d, seed):
  """Test Continuous discretization with NN get_fun."""
  u = create_pinn_field(domain_2d, seed, hidden_dim=10)

  # Test field can be evaluated
  x = jnp.asarray([0.0, 0.0])
  val = u(x)

  assert val is not None
  assert val.shape == (1, )


def test_continuous_field_on_grid(domain_2d, seed):
  """Test PINN field can be evaluated on_grid."""
  u = create_pinn_field(domain_2d, seed, hidden_dim=10)

  # Get grid evaluation
  u_grid = u.on_grid

  assert u_grid is not None
  assert u_grid.shape == (32, 32, 1)


def test_domain_sampler(domain_2d, seed):
  """Test domain.domain_sampler returns correct shape."""
  sampler = domain_2d.domain_sampler
  batch_size = 128

  samples = sampler(seed, batch_size)

  assert samples.shape == (batch_size, len(domain_2d.N))
  # Samples should be within domain bounds
  domain_size = max(domain_2d.N) * max(domain_2d.dx)
  assert jnp.all(jnp.abs(samples) <= domain_size)


def test_boundary_sampler(domain_2d, seed):
  """Test domain.boundary_sampler returns boundary points."""
  sampler = domain_2d.boundary_sampler
  batch_size = 128

  samples = sampler(seed, batch_size)

  assert samples.shape == (batch_size, len(domain_2d.N))

  # At least some samples should be on boundary
  # (at max/min of at least one dimension)
  domain_size = jnp.asarray(domain_2d.N) * jnp.asarray(domain_2d.dx) / 2
  on_boundary = jnp.any(jnp.abs(jnp.abs(samples) - domain_size) < 1e-6, axis=1)
  assert jnp.sum(on_boundary) > 0, "Some samples should be on boundary"


def test_boundary_sampler_1d(seed):
  """Test boundary sampler for 1D domain."""
  domain_1d = Domain((32, ), (1.0, ))
  sampler = domain_1d.boundary_sampler
  batch_size = 64

  samples = sampler(seed, batch_size)

  assert samples.shape == (batch_size, 1)


def test_vmap_on_continuous_field(domain_2d, seed):
  """Test jax.vmap works on Continuous field evaluation."""
  u = create_pinn_field(domain_2d, seed, hidden_dim=10)

  # Create batch of coordinates
  sampler = domain_2d.domain_sampler
  coords = sampler(seed, 32)

  # Use vmap to evaluate field at all coordinates
  @jax.vmap
  def eval_field(x):
    return u(x)

  results = eval_field(coords)

  assert results.shape == (32, 1)


def test_pinn_boundary_loss(domain_2d, seed):
  """Test boundary loss computation (no optimization)."""
  u = create_pinn_field(domain_2d, seed, hidden_dim=10)
  boundary_sampler = domain_2d.boundary_sampler
  batch_size = 64

  def boundary_loss(u, seed, batch_size):
    coords = boundary_sampler(seed, batch_size)

    @jax.vmap
    def eval_field(x):
      return u(x)

    field_vals = eval_field(coords)
    # Loss = mean squared value (for Dirichlet BC: u=0)
    return jnp.mean(field_vals**2)

  loss_val = boundary_loss(u, seed, batch_size)

  assert jnp.ndim(loss_val) == 0
  assert loss_val >= 0


def test_pinn_domain_loss(domain_2d, seed):
  """Test domain loss with residual computation."""
  u = create_pinn_field(domain_2d, seed, hidden_dim=10)
  domain_sampler = domain_2d.domain_sampler
  batch_size = 64

  def domain_loss(u, seed, batch_size):
    coords = domain_sampler(seed, batch_size)

    # Compute Laplacian at sampled points
    @jax.vmap
    def residual(x):
      lap_u = laplacian(u)
      return lap_u(x)

    residuals = residual(coords)
    return jnp.mean(residuals**2)

  loss_val = domain_loss(u, seed, batch_size)

  assert jnp.ndim(loss_val) == 0
  assert loss_val >= 0


def test_pinn_gradient_computation(domain_2d, seed):
  """Test jax.grad works with PINN loss functions."""
  domain_sampler = domain_2d.domain_sampler
  batch_size = 32

  def total_loss(u, seed):
    # Simple domain loss
    coords = domain_sampler(seed, batch_size)

    @jax.vmap
    def residual(x):
      lap_u = laplacian(u)
      return lap_u(x)

    residuals = residual(coords)
    return jnp.mean(residuals**2)

  u = create_pinn_field(domain_2d, seed, hidden_dim=10)

  # Compute gradient
  grad_fn = jax.grad(total_loss)
  grads = grad_fn(u, seed)

  # Check gradients exist and have same pytree structure as u
  assert grads is not None
  assert hasattr(grads, 'params')


def test_pinn_with_initial_conditions(domain_2d_spacetime, seed):
  """Test PINN with initial condition enforcement."""
  u = create_pinn_field(domain_2d_spacetime, seed, hidden_dim=10)
  domain_sampler = domain_2d_spacetime.domain_sampler
  batch_size = 64

  def initial_condition_loss(u, seed, batch_size):
    # Sample from domain
    coords = domain_sampler(seed, batch_size)

    # Set time to t=0 (second coordinate)
    coords_t0 = coords.at[:, 1].set(-0.5 * domain_2d_spacetime.N[1] *
                                    domain_2d_spacetime.dx[1])

    @jax.vmap
    def eval_field(x):
      return u(x)

    field_vals = eval_field(coords_t0)
    # Target: u(x,0) = sin(pi*x)
    target = jnp.sin(jnp.pi * coords[:, 0:1])
    return jnp.mean((field_vals - target)**2)

  loss_val = initial_condition_loss(u, seed, batch_size)

  assert jnp.ndim(loss_val) == 0
  assert loss_val >= 0


def test_pinn_gradient_operator(domain_2d, seed):
  """Test gradient operator works with PINN fields."""
  u = create_pinn_field(domain_2d, seed, hidden_dim=10)

  # Compute gradient
  grad_u = gradient(u)

  # Evaluate gradient at a point
  x = jnp.asarray([0.0, 0.0])
  grad_val = grad_u(x)

  assert grad_val is not None
  assert grad_val.shape == (2, )    # 2D gradient
