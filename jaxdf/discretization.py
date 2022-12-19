import warnings
from typing import Callable, TypeVar

from jax import eval_shape
from jax import numpy as jnp
from jax import vmap
from jax.tree_util import register_pytree_node_class

from jaxdf.core import Field, new_discretization
from jaxdf.geometry import Domain

PyTree = TypeVar("PyTree")


@new_discretization
class Linear(Field):
    r"""This discretization assumes that the field is a linear function of the
    parameters contained in `Linear.params`.
    """

    def __init__(
        self,
        params: PyTree,
        domain: Domain,
        dims=1,
        aux=None,
    ):
        super().__init__(params, domain, dims, aux)


@register_pytree_node_class
class Continuous(Field):
    r"""A continous discretization, which is defined via a `get_field` function stored
    in the `aux` parameters. Its operations are implemented using function composition
    and autograd.

    !!! example
        ```python
        def get_field(params, x):
          return jnp.tanh(params[0] * x + params[1])

        params = jnp.array([1.0, 2.0])
        domain = Domain((16,), (0.1,))
        a = Continuous(params, domain, get_field)
        ```

    """

    def __init__(self, params: PyTree, domain: Domain, get_fun: Callable):
        r"""Initializes a continuous discretization.

        Args:
          params (PyTree): The parameters of the discretization.
          domain (Domain): The domain of the discretization.
          get_fun (Callable): A function that takes a parameter vector and a point in
          the domain and returns the field at that point. The signature of this
          function is `get_field(params, x)`.

        Returns:
          Continuous: A continuous discretization.
        """
        aux = {"get_field": get_fun}
        x = domain.origin
        dims = eval_shape(get_fun, params, x).shape
        super().__init__(params, domain, dims, aux)

    def tree_flatten(self):
        children = (self.params,)
        aux_data = (self.dims, self.domain, self.aux["get_field"])
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params = children[0]
        dims, domain, get_fun = aux_data
        a = cls(params, domain=domain, get_fun=get_fun)
        return a

    def replace_params(self, new_params):
        r"""Replaces the parameters of the discretization with new ones. The domain
        and `get_field` function are not changed.

        Args:
          new_params (PyTree): The new parameters of the discretization.

        Returns:
          Continuous: A continuous discretization with the new parameters.
        """
        return self.__class__(new_params, self.domain, self.aux["get_field"])

    def update_fun_and_params(
        self,
        params: PyTree,
        get_field: Callable,
    ):
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
    def from_function(cls, domain, init_fun: Callable, get_field: Callable, seed):
        r"""Creates a continuous discretization from a `get_field` function.

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
        return self.get_field(x)

    def get_field(self, x):
        r"""Same as `__call__`."""
        return self.aux["get_field"](self.params, x)

    @property
    def on_grid(self):
        """Returns the field on the grid points of the domain."""
        fun = self.aux["get_field"]
        ndims = len(self.domain.N)
        for _ in range(ndims):
            fun = vmap(fun, in_axes=(None, 0))

        return fun(self.params, self.domain.grid)


@register_pytree_node_class
class OnGrid(Linear):
    r"""A linear discretization on the grid points of the domain."""

    def __init__(
        self,
        params: PyTree,
        domain: Domain,
    ):
        r"""Initializes a linear discretization on the grid points of the domain.

        Args:
          params (PyTree): The parameters of the discretization.
          domain (Domain): The domain of the discretization.

        Returns:
          OnGrid: A linear discretization on the grid points of the domain.
        """
        self.domain = domain
        self.params = params

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        # Automatically add a dimension for scalar fields, if possible.
        # See this for an explanation of the first if: https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization
        if not (type(value) is object or value is None or isinstance(value, OnGrid)):
            if len(value.shape) == len(self.domain.N):
                value = jnp.expand_dims(value, -1)
        self._params = value

    @property
    def dims(self):
        return self.params.shape[-1]

    def tree_flatten(self):
        children = (self.params,)
        aux_data = (self.domain,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params = children[0]
        domain = aux_data[0]
        a = cls(params, domain=domain)
        return a

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
        if self.ndim + 1 == len(self.params.shape):
            raise IndexError(
                "Indexing is only supported if there's at least one batch / time dimension"
            )

        new_params = self.params[idx]
        return self.__class__(new_params, self.domain)

    def __repr__(self):
        dims = self.dims
        size = self.domain.N
        return f"{self.__class__.__name__}[dims={dims}, size={size}]"

    @classmethod
    def empty(cls, domain, dims=1):
        r"""Creates an empty OnGrid field (zero field). Equivalent to
        `OnGrid(jnp.zeros(domain.N), domain)`.
        """
        N = tuple(
            list(domain.N)
            + [
                dims,
            ]
        )
        return cls(jnp.zeros(N), domain)

    @property
    def is_field_complex(self) -> bool:
        r"""Checks if a field is complex.

        Returns:
          bool: Whether the field is complex.
        """
        return self.params.dtype == jnp.complex64 or self.params.dtype == jnp.complex128

    @property
    def real(self) -> bool:
        r"""Checks if a field is real.

        Returns:
          bool: Whether the field is real.
        """
        return not self.is_field_complex

    @classmethod
    def from_grid(cls, grid_values, domain):
        r"""Creates an OnGrid field from a grid of values.

        Args:
          grid_values (ndarray): The grid of values.
          domain (Domain): The domain of the discretization.
        """
        return cls(grid_values, domain)

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


@register_pytree_node_class
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

        k_vec = [jnp.exp(-1j * k * delta) for k, delta in zip(self._freq_axis, shift)]
        ffts = self._ffts

        new_params = self.params

        def single_shift(axis, u):
            u = jnp.moveaxis(u, axis, -1)
            Fx = ffts[0](u, axis=-1)
            iku = Fx * k_vec[axis]
            du = ffts[1](iku, axis=-1, n=u.shape[-1])
            return jnp.moveaxis(du, -1, axis)

        for ax in range(self.ndim):
            new_params = single_shift(ax, new_params)

        origin = tuple([0] * self.ndim)
        return new_params[origin]

    def get_field(self, x):
        warnings.warn(
            "FourierSeries.get_field is deprecated. Use FourierSeries.__call__ instead.",
            DeprecationWarning,
        )
        return self.__call__(x)

    @property
    def _freq_axis(self):
        r"""Returns the frequency axis of the grid"""
        if self.is_field_complex:

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
        if self.real:
            return [jnp.fft.rfft, jnp.fft.irfft]
        else:
            return [jnp.fft.fft, jnp.fft.ifft]

    @property
    def _cut_freq_axis(self):
        r"""Same as _freq_axis, but last frequency axis is relative to a real FFT.
        Those frequency axis match with the ones of the rfftn function
        """

        def f(N, dx):
            return jnp.fft.fftfreq(N, dx) * 2 * jnp.pi

        k_axis = [f(n, delta) for n, delta in zip(self.domain.N, self.domain.dx)]
        if not self.is_field_complex:
            k_axis[-1] = (
                jnp.fft.rfftfreq(self.domain.N[-1], self.domain.dx[-1]) * 2 * jnp.pi
            )
        return k_axis

    @property
    def _cut_freq_grid(self):
        return jnp.stack(jnp.meshgrid(*self._cut_freq_axis, indexing="ij"), axis=-1)

    @property
    def _freq_grid(self):
        return jnp.stack(jnp.meshgrid(*self._freq_axis, indexing="ij"), axis=-1)


@register_pytree_node_class
class FiniteDifferences(OnGrid):
    r"""A Finite Differences field defined on a collocation grid."""

    def __init__(
        self,
        params,
        domain,
        accuracy=8,
    ):
        r"""Initializes a Finite Differences field on a collocation grid.
        Args:
          params (PyTree): The parameters of the discretization.
          domain (Domain): The domain of the discretization.
        Returns:
          FiniteDifferences: A Finite Differences field on a collocation grid.
        """
        self.domain = domain
        self.params = params
        self.accuracy = accuracy

    def tree_flatten(self):
        children = (self.params,)
        aux_data = (self.domain, self.accuracy)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params = children[0]
        domain, accuracy = aux_data
        a = cls(params, domain=domain, accuracy=accuracy)
        return a
