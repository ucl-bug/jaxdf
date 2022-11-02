from jaxdf import FourierSeries
from jaxdf.geometry import Domain

from jax import numpy as jnp
import jax


def test_vmap_over_ongrid():
    # Checks for this:
    # https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization
    def foo(field, x):
        return field

    vfoo = jax.vmap(foo, in_axes=(None, 0))

    N = (8,8)
    params = jnp.ones(N)
    domain = Domain(N, (1,1))

    fs = FourierSeries(params, domain)
    values = jnp.asarray([1,2,3,4])

    print(vfoo(fs, values))