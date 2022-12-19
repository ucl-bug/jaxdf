from jax import grad, jit
from jax import numpy as jnp

from jaxdf import FourierSeries, operator
from jaxdf import operators as jops
from jaxdf.geometry import Domain


def test_readme_example():
    # Defining operator
    @operator
    def custom_op(u, *, params=None):
        grad_u = jops.gradient(u)
        diag_jacobian = jops.diag_jacobian(grad_u)
        laplacian = jops.sum_over_dims(diag_jacobian)
        sin_u = jops.compose(u)(jnp.sin)
        return laplacian + sin_u

    # Defining discretizations
    domain = Domain((128, 128), (1.0, 1.0))
    parameters = jnp.ones((128, 128, 1))
    u = FourierSeries(parameters, domain)

    # Define a differentiable loss function
    @jit
    def loss(u):
        v = custom_op(u)
        return jnp.mean(jnp.abs(v.on_grid) ** 2)

    gradient = grad(loss)(u)
    assert gradient.on_grid.shape == (128, 128, 1)
    assert type(gradient) == FourierSeries


if __name__ == "__main__":
    test_readme_example()
