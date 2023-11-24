import inspect
import logging
import types
import warnings
from functools import wraps
from typing import Callable, Union
from warnings import warn

from jaxtyping import PyTree
from plum import Dispatcher

from jaxdf.signatures import (SignatureError, add_defaults,
                              check_eval_init_signatures, check_fun_has_params)

from .geometry import Domain
from .logger import logger, set_logging_level
from .mods import Module

# Initialize the dispatch table
_jaxdf_dispatch = Dispatcher()


# Configuration. This is just for backward compatibility
class _DebugDict(dict):

  def __setitem__(self, __key, __value):
    if __key == "debug_dispatch":
      warn(
          "debug_dispatch is deprecated. Set the logger level to DEBUG instead.",
          DeprecationWarning)
      # Assuming you want to set the logger level based on this value
      if __value:
        set_logging_level(logging.DEBUG)
      else:
        set_logging_level(logging.INFO)
      super().__setitem__(__key, __value)
    else:
      raise ValueError("Only debug_dispatch is supported for now")


debug_config = _DebugDict()


def _abstract_operator(evaluate):
  f = _jaxdf_dispatch.abstract(evaluate)
  return f


def _operator(evaluate, precedence, init_params):
  check_fun_has_params(evaluate)

  # If the parameter initialization function is not provided, then
  # assume that the operator has no parameters
  if init_params is None:

    def init_params(*args, **kwargs):
      return None

  # Verify that the init_params function does not have defaults, as they are inherited from
  # the operator function, so they are not needed and can cause ambiguity
  sig_init_params = inspect.signature(init_params)
  defaults = {
      k: v.default
      for k, v in sig_init_params.parameters.items()
      if v.default is not inspect.Parameter.empty
  }
  if len(defaults) > 0:
    raise SignatureError(
        f"The init_params function {init_params.__name__} must not have default values, as they are inherited from the operator function {evaluate.__name__}. The init_params function has the following default values: {defaults}"
    )

  # Check that the init_params function is compatible with the evaluate function
  check_eval_init_signatures(evaluate, init_params)

  # Create the operator function
  @wraps(evaluate)
  def wrapper(*args, **kwargs):
    # Check if the parameters are not passed
    if "params" not in kwargs:
      # Generate them
      kwargs = add_defaults(evaluate, kwargs, skip=["params"])
      kwargs["params"] = init_params(*args, **kwargs)

    # Log dispatch message
    logger.debug(
        f"Dispatching {evaluate.__name__} with for types {evaluate.__annotations__}"
    )

    outs = evaluate(*args, **kwargs)
    if isinstance(outs, tuple) and len(outs) > 1:
      logger.warning(
          f"Deprecation: Currently only the first output of an operator is considered. This will change in a future release. If you need to return multiple outputs, please return a tuple and a None value, for example: ((out1, out2), None). This happened for the operator `{evaluate.__name__}`."
      )
      # Overload the field class with an extra attribute
      field = outs[0]
    else:
      field = outs

    return field

  # Adds the parameters initializer to the functin wrapper
  wrapper._initialize_parameters = init_params

  # Register the operator in the dispatch table
  logger.debug(f"Registering {evaluate.__name__} with precedence {precedence}")
  f = _jaxdf_dispatch(wrapper, precedence=precedence)

  # Bind an default_params method that returns the default parameters
  def _bound_init_params(self, *args, **kwargs):
    # the method is resolved only on non-keyword arguments,
    # see: https://github.com/wesselb/plum/issues/40#issuecomment-1321164488
    self._resolve_pending_registrations()
    # sig_types = tuple(map(type, args))

    method, _ = self.resolve_method(args)
    kwargs = add_defaults(method, kwargs, skip=["params"])
    return method._initialize_parameters(*args, **kwargs)

  f.default_params = types.MethodType(_bound_init_params, f)

  return f


class Operator:

  def __call__(
      self,
      evaluate: Union[Callable, None] = None,
      init_params: Union[Callable, None] = None,
      precedence: int = 0,
  ):
    if evaluate is None:
      # Returns the decorator
      def decorator(evaluate):
        return _operator(evaluate, precedence, init_params)

      return decorator
    else:
      return _operator(evaluate, precedence, init_params)

  def abstract(self, evaluate: Callable):
    """Decorator for defining abstract operators. This is mainly used
        to define generic docstrings."""
    return _abstract_operator(evaluate)


operator = Operator()
r"""Decorator for defining operators using multiple dispatch. The type annotation of the
    `evaluate` function are used to determine the dispatch rules. The dispatch syntax is the
    same as the Julia one, that is: operators are dispatched on the types of the positional arguments.

    Args:
      evaluate (Callable): A function with the signature `evaluate(field, *args, **kwargs, params)`.
          It must return a tuple, with the first element being a field and the second
          element being the default parameters for the operator.
      init_params (Callable): A function that overrides the default parameters initializer for the
          operator. Useful when running the operator just to get the parameters is expensive.
      precedence (int): The precedence of the operator if an ambiguous match is found.

    Returns:
      Callable: The operator function with signature `evaluate(field, *args, **kwargs, params)`.

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

    """


def discretization(cls):
  warn(
      "jaxdf.discretization is deprecated since the discretization API has been moved to equinox. You don't need this decorator anymore. It iwll now simply act as a pass-through.",
      DeprecationWarning)
  return cls


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


class Field(Module):
  params: PyTree
  domain: Domain

  # For concise code
  @property
  def θ(self):
    r"""Handy alias for the `params` attribute"""
    return self.params

  def __call__(self, x):
    r"""
        An Field can be called as a function, returning the field at a
        desired point.

        !!! example
            ```python
            ...
            a = Continuous.from_function(init_params, domain, get_field)
            field_at_x = a(1.0)
            ```
        """
    raise NotImplementedError(
        f"Not implemented for {self.__class__.__name__} discretization")

  @property
  def on_grid(self):
    """Returns the field on the grid points of the domain."""
    raise NotImplementedError(
        f"Not implemented for {self.__class__.__name__} discretization")

  @property
  def dims(self):
    r"""The dimension of the field values"""
    raise NotImplementedError

  @property
  def is_complex(self) -> bool:
    r"""Checks if a field is complex.

        Returns:
          bool: Whether the field is complex.
        """
    raise NotImplementedError

  @property
  def is_field_complex(self) -> bool:
    warnings.warn(
        "Field.is_field_complex is deprecated. Use Field.is_complex instead.",
        DeprecationWarning,
    )
    return self.is_complex

  @property
  def is_real(self) -> bool:
    return not self.is_complex

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
    return self.__class__(new_params, self.domain)

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
      f"Function not implemented for {type(self)} and {type(other)}")


@operator
def __radd__(self, other, *, params=None):
  raise NotImplementedError(
      f"Function not implemented for {type(self)} and {type(other)}")


@operator
def __sub__(self, other, *, params=None):
  raise NotImplementedError(
      f"Function not implemented for {type(self)} and {type(other)}")


@operator
def __rsub__(self, other, *, params=None):
  raise NotImplementedError(
      f"Function not implemented for {type(self)} and {type(other)}")


@operator
def __mul__(self, other, *, params=None):
  raise NotImplementedError(
      f"Function not implemented for {type(self)} and {type(other)}")


@operator
def __rmul__(self, other, *, params=None):
  raise NotImplementedError(
      f"Function not implemented for {type(self)} and {type(other)}")


@operator
def __neg__(self, *, params=None):
  raise NotImplementedError(f"Function not implemented for {type(self)}")


@operator
def __pow__(self, other, *, params=None):
  raise NotImplementedError(
      f"Function not implemented for {type(self)} and {type(other)}")


@operator
def __rpow__(self, other, *, params=None):
  raise NotImplementedError(
      f"Function not implemented for {type(self)} and {type(other)}")


@operator
def __truediv__(self, other, *, params=None):
  raise NotImplementedError(
      f"Function not implemented for {type(self)} and {type(other)}")


@operator
def __rtruediv__(self, other, *, params=None):
  raise NotImplementedError(
      f"Function not implemented for {type(self)} and {type(other)}")
