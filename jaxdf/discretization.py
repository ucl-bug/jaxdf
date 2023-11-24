import warnings
from typing import Callable, TypeVar

import equinox as eqx
from equinox import tree_equal
from jax import eval_shape
from jax import numpy as jnp
from jax import vmap
from jaxtyping import PyTree

from jaxdf.core import Field
from jaxdf.geometry import Domain

PyTree = TypeVar("PyTree")


class Linear(Field):
  r"""This discretization assumes that the field is a linear function of the
    parameters contained in `Linear.params`.
    """
  params: PyTree
  domain: Domain

  def __eq__(self, other):
    return tree_equal(self, other) * (self.domain == other.domain)

  @property
  def is_complex(self) -> bool:
    r"""Checks if a field is complex.

        Returns:
          bool: Whether the field is complex.
        """
    return self.params.dtype == jnp.complex64 or self.params.dtype == jnp.complex128


class Continuous(Field):
  r"""A continuous discretization. This discretization assumes that the field is a
    function of the parameters contained in `Continuous.params` and a point in the
    domain. The function that computes the field from the parameters and the point in
    the domain is contained in `Continuous.get_fun`. This function has the signature
    `get_fun(params, x)`, where `params` is the parameter vector and `x` is the point
    in the domain."""
  params: PyTree
  domain: Domain
  get_fun: Callable = eqx.field(static=True)

  def __eq__(self, other):
    return tree_equal(self, other) * (self.get_fun == other.get_fun)

  @property
  def is_complex(self) -> bool:
    r"""Checks if a field is complex.

        Returns:
          bool: Whether the field is complex.
        """
    origin = self.domain.origin
    value = self.get_fun(self.params, origin)
    return jnp.iscomplexobj(value)

  @property
  def dims(self):
    x = self.domain.origin
    return eval_shape(self.get_fun, self.params, x).shape

  def replace_params(self, new_params: PyTree) -> "Continuous":
    r"""Replaces the parameters of the discretization with new ones. The domain
        and `get_field` function are not changed.

        Args:
          new_params (PyTree): The new parameters of the discretization.

        Returns:
          Continuous: A continuous discretization with the new parameters.
        """
    return self.__class__(new_params, self.domain, self.get_field)

  def update_fun_and_params(
      self,
      params: PyTree,
      get_field: Callable,
  ) -> "Continuous":
    r"""Updates the parameters and the function of the discretization.

        Args:
          params (PyTree): The new parameters of the discretization.
          get_field (Callable): A function that takes a parameter vector and a point in
            the domain and returns the field at that point. The signature of this
            function is `get_field(params, x)`.

        Returns:
          Continuous: A continuous discretization with the new parameters and function.
        """
    return self.__class__(params, self.domain, get_field)

  @classmethod
  def from_function(cls, domain, init_fun: Callable, get_field: Callable,
                    seed):
    r"""Creates a continuous discretization from an `init_fun` function.

        Args:
          domain (Domain): The domain of the discretization.
          init_fun (Callable): A function that initializes the parameters of the
            discretization. The signature of this function is `init_fun(rng, domain)`.
          get_field (Callable): A function that takes a parameter vector and a point in
            the domain and returns the field at that point. The signature of this
            function is `get_field(params, x)`.
          seed (int): The seed for the random number generator.

        Returns:
          Continuous: A continuous discretization.
        """
    params = init_fun(seed, domain)
    return cls(params, domain=domain, get_fun=get_field)

  def __call__(self, x):
    r"""
        An object of this class can be called as a function, returning the field at a
        desired point.

        !!! example
            ```python
            ...
            a = Continuous.from_function(init_params, domain, get_field)
            field_at_x = a(1.0)
            ```
        """
    return self.get_fun(self.params, x)

  def get_field(self, x):
    warnings.warn(
        "Continuous.get_field is deprecated. Use Continuous.__call__ instead.",
        DeprecationWarning,
    )
    return self.__call__(x)

  @property
  def on_grid(self):
    """Returns the field on the grid points of the domain."""
    fun = self.get_fun
    ndims = len(self.domain.N)
    for _ in range(ndims):
      fun = vmap(fun, in_axes=(None, 0))

    return fun(self.params, self.domain.grid)


class OnGrid(Linear):

  def __check_init__(self):
    # Check if the number of dimensions of the parameters is correct, fix if needed
    if len(self.params.shape) == len(self.domain.N):
      # If only the last one is missing, add it
      if self.params.shape == self.domain.N:
        object.__setattr__(self, "params", jnp.expand_dims(self.params,
                                                           axis=-1))

    if self.params.shape == self.domain.N:
      raise ValueError(
          f"The number of dimensions of the parameters is incorrect. It should be the number of dimensions of the domain plus at least one more. The parameters have shape {self.params.shape} and the domain has shape {self.domain.N}"
      )

  def add_dim(self):
    """Adds a dimension at the end of the `params` array."""
    new_params = jnp.expand_dims(self.params, axis=-1)
    # Returns a new object
    return self.__class__(new_params, self.domain)

  @property
  def dims(self):
    return self.params.shape[-1]

  def __getitem__(self, idx):
    r"""Allow indexing when leading batch / time dimensions are
        present in the parameters

        !!! example
            ```python
            ...
            domain = Domain((16, (1.0,))

            # 10 fields
            params = random.uniform(key, (10, 16, 1))
            a = OnGrid(params, domain)

            # Field at the 5th index
            field = a[5]
            ```

        Returns:
          OnGrid: A linear discretization on the grid points of the domain.

        Raises:
          IndexError: If the field is not indexable (single field).
        """
    if self.domain.ndim + 1 == len(self.params.shape):
      raise IndexError(
          "Indexing is only supported if there's at least one batch / time dimension"
      )

    new_params = self.params[idx]
    return self.__class__(new_params, self.domain)

  @classmethod
  def empty(cls, domain, dims=1):
    r"""Creates an empty OnGrid field (zero field). Equivalent to
        `OnGrid(jnp.zeros(domain.N), domain)`.
        """
    _N = list(domain.N) + [dims]
    N = tuple(_N)
    return cls(jnp.zeros(N), domain)

  @classmethod
  def from_grid(cls, grid_values, domain):
    r"""Creates an OnGrid field from a grid of values.

        Args:
          grid_values (ndarray): The grid of values.
          domain (Domain): The domain of the discretization.
        """
    a = cls(grid_values, domain)
    return a

  def replace_params(self, new_params):
    r"""Replaces the parameters of the discretization with new ones. The domain
        is not changed.

        Args:
          new_params (PyTree): The new parameters of the discretization.

        Returns:
          OnGrid: A linear discretization with the new parameters.
        """
    return self.__class__(new_params, self.domain)

  @property
  def on_grid(self):
    r"""The field on the grid points of the domain."""
    return self.params


class FourierSeries(OnGrid):
  r"""A Fourier series field defined on a collocation grid."""

  def __call__(self, x):
    r"""Uses the Fourier shift theorem to compute the value of the field
        at an arbitrary point. Requires N*2 one dimensional FFTs.

        Args:
          x (float): The point at which to evaluate the field.

        Returns:
          float, jnp.ndarray: The value of the field at the point.
        """
    dx = jnp.asarray(self.domain.dx)
    domain_size = jnp.asarray(self.domain.N) * dx
    shift = x - domain_size / 2 + 0.5 * dx

    k_vec = [
        jnp.exp(-1j * k * delta) for k, delta in zip(self._freq_axis, shift)
    ]
    ffts = self._ffts

    new_params = self.params

    def single_shift(axis, u):
      u = jnp.moveaxis(u, axis, -1)
      Fx = ffts[0](u, axis=-1)
      iku = Fx * k_vec[axis]
      du = ffts[1](iku, axis=-1, n=u.shape[-1])
      return jnp.moveaxis(du, -1, axis)

    for ax in range(self.domain.ndim):
      new_params = single_shift(ax, new_params)

    origin = tuple([0] * self.domain.ndim)
    return new_params[origin]

  @property
  def _freq_axis(self):
    r"""Returns the frequency axis of the grid"""
    if self.is_complex:

      def f(N, dx):
        return jnp.fft.fftfreq(N, dx) * 2 * jnp.pi

    else:

      def f(N, dx):
        return jnp.fft.rfftfreq(N, dx) * 2 * jnp.pi

    k_axis = [f(n, delta) for n, delta in zip(self.domain.N, self.domain.dx)]
    return k_axis

  @property
  def _ffts(self):
    r"""Returns the FFT and iFFT functions that are appropriate for the
        field type (real or complex)
        """
    if self.is_real:
      return [jnp.fft.rfft, jnp.fft.irfft]
    else:
      return [jnp.fft.fft, jnp.fft.ifft]

  @property
  def _cut_freq_axis(self):
    r"""Same as `FourierSeries._freq_axis`, but last frequency axis is relative to a real FFT.
        Those frequency axis match with the ones of the rfftn function
        """

    def f(N, dx):
      return jnp.fft.fftfreq(N, dx) * 2 * jnp.pi

    k_axis = [f(n, delta) for n, delta in zip(self.domain.N, self.domain.dx)]
    if not self.is_complex:
      k_axis[-1] = (jnp.fft.rfftfreq(self.domain.N[-1], self.domain.dx[-1]) *
                    2 * jnp.pi)
    return k_axis

  @property
  def _cut_freq_grid(self):
    return jnp.stack(jnp.meshgrid(*self._cut_freq_axis, indexing="ij"),
                     axis=-1)

  @property
  def _freq_grid(self):
    return jnp.stack(jnp.meshgrid(*self._freq_axis, indexing="ij"), axis=-1)


class FiniteDifferences(OnGrid):
  r"""A Finite Differences field defined on a collocation grid."""
  params: PyTree
  domain: Domain
  accuracy: int = eqx.field(default=8, static=True)
