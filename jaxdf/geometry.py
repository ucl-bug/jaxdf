from functools import reduce
from typing import Iterable

import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random

from .mods import Module


class Domain(Module):
  r"""Domain class describing a rectangular domain

    Attributes:
        size (Tuple[int]): The size of the domain in absolute units.
        dx (Tuple(float)): The unit of measure
    """
  N: Iterable[int] = eqx.field(default=(32, 32), static=True)
  dx: Iterable[float] = eqx.field(default=(1.0, 1.0), static=True)

  @property
  def size(self):
    r"""The lenght of the grid sides

        Returns:
            Tuple[float]: The size of the domain, in absolute units

        """
    return list(map(lambda x: x[0] * x[1], zip(self.N, self.dx)))

  @property
  def ndim(self):
    r"""The number of dimensions of the domain

        Returns:
            int: The number of dimensions of the domain

        """
    return len(self.N)

  @property
  def cell_volume(self):
    r"""The volume of a single cell

        Returns:
            float: The volume of a single cell

        """
    return reduce(lambda x, y: x * y, self.dx)

  @property
  def spatial_axis(self):
    r"""The spatial axis of the domain

        Returns:
            Tuple[jnp.array]: The spatial axis of the domain

        """

    def _make_axis(n, delta):
      if n % 2 == 0:
        return jnp.arange(0, n) * delta - delta * n / 2
      else:
        return jnp.arange(0, n) * delta - delta * (n - 1) / 2

    axis = [_make_axis(n, delta) for n, delta in zip(self.N, self.dx)]
    axis = [ax - jnp.mean(ax) for ax in axis]
    return axis

  @property
  def boundary_sampler(self):
    r"""Returns a function that samples a point on the boundary of the domain

        Returns:
            Callable: A function that samples a point on the boundary of the domain.
                This function takes a seed and an integer number of samples and returns
                a list of samples.

        !!! example
            ```python
            >>> domain = Domain((10, 10), (0.1, 0.1))
            >>> boundary_sampler = domain.boundary_sampler
            >>> boundary_sampler(random.PRNGKey(0), 10)
            Array([[-0.02072918,  0.5       ],
                   [-0.5       ,  0.49063694],
                   [-0.18872023, -0.5       ],
                   [ 0.31801188, -0.5       ],
                   [-0.1319474 , -0.5       ],
                   [ 0.5       , -0.36944878],
                   [ 0.5       ,  0.46956718],
                   [ 0.4608934 , -0.5       ],
                   [-0.09031796,  0.5       ],
                   [-0.5       ,  0.40659428]], dtype=float32)

            ```

        """
    L = jnp.asarray(self.size) / 2

    def sample(seed):
      seeds = random.split(seed, 3)
      first = 2 * jnp.expand_dims(random.uniform(seeds[0]), 0) - 1
      others = 2 * random.bernoulli(seeds[1], shape=(self.ndim - 1, )) - 1
      sample = jnp.concatenate([first, others]).astype(jnp.float32)
      random_perm = random.permutation(seeds[2], sample)
      sample = random_perm * L
      return sample

    def multi_samples(seed, num_samples: int):
      seeds = random.split(seed, num_samples)
      return jax.vmap(sample)(seeds)

    return multi_samples

  @property
  def domain_sampler(self):
    r"""Returns a function that samples a point in the domain

        Returns:
            Callable: A function that samples a point in the domain.
                This function takes a seed and an integer number of samples and returns
                a list of samples.

        !!! example
            ```python
            >>> domain = Domain((10, 10), (0.1, 0.1))
            >>> domain_sampler = domain.domain_sampler
            >>> domain_sampler(random.PRNGKey(0), 10)
            Array([[ 0.06298566,  0.35970068],
                   [-0.20049798,  0.05455852],
                   [ 0.33402848, -0.04824698],
                   [ 0.27945423,  0.2805649 ],
                   [ 0.49464726,  0.3473643 ],
                   [-0.16299951, -0.27665186],
                   [-0.06442916,  0.04995835],
                   [ 0.05011427, -0.17267668],
                   [-0.39805043, -0.05386746],
                   [ 0.46900105,  0.21520817]], dtype=float32)

            ```

        """
    L = jnp.asarray(self.size) / 2

    def sample(seed):
      sample = 2 * random.uniform(seed, shape=(self.ndim, )) - 1
      return sample * L

    def multi_samples(seed, num_samples: int):
      seeds = random.split(seed, num_samples)
      return jax.vmap(sample)(seeds)

    return multi_samples

  @property
  def origin(self):
    return jnp.zeros((self.ndim, ))

  @staticmethod
  def _make_grid_from_axis(axis):
    return jnp.stack(jnp.meshgrid(*axis, indexing="ij"), axis=-1)

  @property
  def grid(self):
    """A grid of spatial position, of size
        `Nx x Ny x Nz x ... x num_axis` such that the element
        `[x1,x2,x3, .., :]` is a coordinate vector.

        Returns:
            jnp.array: A grid of spatial position

        """
    axis = self.spatial_axis
    return self._make_grid_from_axis(axis)
