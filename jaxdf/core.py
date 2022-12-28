import types
from functools import wraps
from typing import Callable, Union

from jax.tree_util import register_pytree_node_class, tree_map
from plum import Dispatcher, type_of

from jaxdf.exceptions import check_fun_has_params

# Initialize the dispatch table
_jaxdf_dispatch = Dispatcher()

# Configuration
debug_config = {
    "debug_dispatch": False,
}


def _operator(evaluate, precedence, init_params):
    check_fun_has_params(evaluate)

    # If the parameter initialization function is not provided, then
    # assume that the operator has no parameters
    if init_params is None:

        def init_params(*args, **kwargs):
            return None

    # Create the operator function
    @wraps(evaluate)
    def wrapper(*args, **kwargs):
        # Check if the parameters are not passed
        if "params" not in kwargs:
            kwargs["params"] = init_params(*args, **kwargs)

        if debug_config["debug_dispatch"]:
            print(
                f"Dispatching {evaluate.__name__} with for types {evaluate.__annotations__}"
            )

        outs = evaluate(*args, **kwargs)
        if isinstance(outs, tuple) and len(outs) > 1:
            # Overload the field class with an extra attribute
            field = outs[0]
        else:
            field = outs

        return field

    # Adds the parameters initializer to the functin wrapper
    wrapper._initialize_parameters = init_params

    # Register the operator in the dispatch table
    f = _jaxdf_dispatch(wrapper, precedence=precedence)

    # Bind an default_params method that returns the default parameters
    def _bound_init_params(self, *args, **kwargs):
        # the method is resolved only on non-keyword arguments,
        # see: https://github.com/wesselb/plum/issues/40#issuecomment-1321164488
        self._resolve_pending_registrations()
        sig_types = tuple([type_of(x) for x in args])

        # """ # TODO: This is a hack that shouldn't be needed
        # if not self._runtime_type_of:
        #    sig_types = tuple([type(x) for x in args])
        # else:
        #    sig_types = tuple([promised_type_of.resolve()(x) for x in args])
        # """

        method, _ = self.resolve_method(*sig_types)
        return method._initialize_parameters(*args, **kwargs)

    f.default_params = types.MethodType(_bound_init_params, f)

    return f


def operator(
    evaluate: Union[Callable, None] = None,
    init_params: Union[Callable, None] = None,
    precedence: int = 0,
):
    r"""Decorator for defining operators using multiple dispatch. The type annotation of the
    `evaluate` function are used to determine the dispatch rules. The dispatch syntax is the
    same as the Julia one, that is: operators are dispatched on the types of the positional arguments.
    Keyword arguments are not considered for dispatching.

    Keyword arguments are defined after the `*` in the function signature.

    !!! example
        ```python
        @operator
        def my_operator(x: FourierSeries, *, dx: float, params=None):
          ...
        ```

    The argument `params` is mandatory and it must be a keyword argument. It is used to pass the
    parameters of the operator, for example the stencil coefficients of a finite difference operator.

    The default value of the parameters is specified by the `init_params` function, as follows:

    !!! example
        ```python

        def params_initializer(x, *, dx):
          return {"stencil": jnp.ones(x.shape) * dx}

        @operator(init_params=params_initializer)
        def my_operator(x, *, dx, params=None):
          b = params["stencil"] / dx
          y_params = jnp.convolve(x.params, b, mode="same")
          return x.replace_params(y_params)
        ```

    The default value of `params` is not considered during computation.
    If the operator has no parameters, the `init_params` function can be omitted. In this case, the
    `params` value is set to `None`.

    For constant parameters, the `constants` function can be used:

    !!! example
        ```python
        @operator(init_params=constants({"a": 1, "b": 2.0}))
        def my_operator(x, *, params):
          return x + params["a"] + params["b"]
        ```


    Args:
      evaluate (Callable): A function with the signature `evaluate(field, *args, **kwargs, params)`.
        It must return a tuple, with the first element being a field and the second
        element being the default parameters for the operator.
      init_params (Callable): A function that overrides the default parameters initializer for the
        operator. Useful when running the operator just to get the parameters is expensive.
      precedence (int): The precedence of the operator if an ambiguous match is found.

    Returns:
      Callable: The operator function with signature `evaluate(field, *args, **kwargs, params)`.

    """
    if evaluate is None:
        # Returns the decorator
        def decorator(evaluate):
            return _operator(evaluate, precedence, init_params)

        return decorator
    else:
        return _operator(evaluate, precedence, init_params)


def new_discretization(cls):
    r"""Wrapper around `jax.tree_util.register_pytree_node_class` that can
    be used to register a new discretization.

    If the discretization doesn't have the same `__init__` function as the
    parent class, the methods `tree_flatten` and `tree_unflatten` must be
    present (see [Extending pytrees](https://jax.readthedocs.io/en/latest/pytrees.html)
    in the JAX documentation).

    !!! example
        ```python
        @new_discretization
        class Polynomial(Continuous):
          @classmethod
          def from_params(cls, params, domain):
            def get_fun(params, x):
              i = jnp.arange(len(params))
              powers = x**i
              return jnp.expand_dims(jnp.sum(params*(x**i)), -1)
            return cls(params, domain, get_fun)
        ```
    """
    return register_pytree_node_class(cls)


def constants(value) -> Callable:
    r"""This is a higher order function for defining constant parameters of
    operators, independent of the operator arguments.

    !!! example

        ```python
        @operator(init_params=constants({"a": 1, "b": 2.0}))
        def my_operator(x, *, params):
          return x + params["a"] + params["b"]
        ```

    Args:
      value (Any): The value of the constant parameters.

    Returns:
      Callable: The parameters initializer function that returns the constant value.
    """

    def init_params(*args, **kwargs):
        return value

    return init_params


@new_discretization
class Field(object):
    r"""The base-class for all discretizations. This class is also responsible
    for binding the operators in `jaxdf.operators.magic` to
    the magic methods of the discretization.

    Normally you should not use this class directly, but instead use the
    `new_discretization` decorator to register
    a new discretization based on this class.
    """

    def __init__(
        self,
        params,
        domain,
        dims=1,
        aux=None,
    ):
        self.params = params
        self.domain = domain
        self.dims = dims
        self.aux = aux

    def tree_flatten(self):
        children = (self.params,)
        aux_data = (self.dims, self.domain, self.aux)
        return (children, aux_data)

    @property
    def ndim(self):
        return self.domain.ndim

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params = children[0]
        dims, domain, aux = aux_data
        a = cls(params, dims=dims, domain=domain, aux=aux)
        return a

    def __repr__(self):  #
        classname = self.__class__.__name__
        return f"Field {classname}"

    def __str__(self):
        return self.__repr__()

    def replace_params(self, new_params):
        r"""Returns a new field of the same type, with the same domain and auxiliary data, but with new parameters.

        !!! example
            ```python
            x = FourierSeries(jnp.ones(10), domain=domain)
            y_params = x.params + 1
            y = x.replace_params(y_params)
            ```

        Args:
          new_params (Any): The new parameters.

        Returns:
          Field: A new field with the same domain and auxiliary data, but with new parameters.
        """
        return self.__class__(new_params, self.domain, self.dims, self.aux)

    # Dummy magic functions to make it work with
    # the dispatch system
    def __add__(self, other):
        return __add__(self, other)

    def __radd__(self, other):
        return __radd__(self, other)

    def __sub__(self, other):
        return __sub__(self, other)

    def __rsub__(self, other):
        return __rsub__(self, other)

    def __mul__(self, other):
        return __mul__(self, other)

    def __rmul__(self, other):
        return __rmul__(self, other)

    def __neg__(self):
        return __neg__(self)

    def __pow__(self, other):
        return __pow__(self, other)

    def __rpow__(self, other):
        return __rpow__(self, other)

    def __truediv__(self, other):
        return __truediv__(self, other)

    def __rtruediv__(self, other):
        return __rtruediv__(self, other)


@operator
def __add__(self, other, *, params=None):
    raise NotImplementedError(
        f"Function not implemented for {type(self)} and {type(other)}"
    )


@operator
def __radd__(self, other, *, params=None):
    raise NotImplementedError(
        f"Function not implemented for {type(self)} and {type(other)}"
    )


@operator
def __sub__(self, other, *, params=None):
    raise NotImplementedError(
        f"Function not implemented for {type(self)} and {type(other)}"
    )


@operator
def __rsub__(self, other, *, params=None):
    raise NotImplementedError(
        f"Function not implemented for {type(self)} and {type(other)}"
    )


@operator
def __mul__(self, other, *, params=None):
    raise NotImplementedError(
        f"Function not implemented for {type(self)} and {type(other)}"
    )


@operator
def __rmul__(self, other, *, params=None):
    raise NotImplementedError(
        f"Function not implemented for {type(self)} and {type(other)}"
    )


@operator
def __neg__(self, *, params=None):
    raise NotImplementedError(f"Function not implemented for {type(self)}")


@operator
def __pow__(self, other, *, params=None):
    raise NotImplementedError(
        f"Function not implemented for {type(self)} and {type(other)}"
    )


@operator
def __rpow__(self, other, *, params=None):
    raise NotImplementedError(
        f"Function not implemented for {type(self)} and {type(other)}"
    )


@operator
def __truediv__(self, other, *, params=None):
    raise NotImplementedError(
        f"Function not implemented for {type(self)} and {type(other)}"
    )


@operator
def __rtruediv__(self, other, *, params=None):
    raise NotImplementedError(
        f"Function not implemented for {type(self)} and {type(other)}"
    )

    # Lifted jax functions for convenience


def params_map(f: Callable, field: Field, *rest) -> Field:
    r"""Maps a function to the parameters of a Field.

    Since a Field is a pytree, this is equivalent to (and implemented
    using) `jax.tree_util.tree_map`

    Returns a field with the same type of `f`, with updated
    parameters

    !!! danger
        This function has been deprecated. Use `jaxdf.operators.functions.compose` instead

    Args:
      f (Callable): A function that is applied to all the leaves of the
        parameters of the input fields
      field (Field): The field to map the function to
      *rest: Optional additional fields to map the function to

    Returns:
      Field: A field with the same discretization as `field`, with updated parameters.
    """
    return tree_map(f, field, *rest)
