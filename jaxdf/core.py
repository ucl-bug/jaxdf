
from distutils.log import debug
import inspect
from functools import partial, wraps
from typing import Callable

from jax.tree_util import register_pytree_node, register_pytree_node_class, tree_map, tree_multimap
from numpy import issubdtype

from plum import Dispatcher
from setuptools import setup

from jaxdf import util

Params = None
_jaxdf_dispatch = Dispatcher()
debug_config ={
  "debug_dispatch": False,
}

def _operator(evaluate, precedence):
  @wraps(evaluate)
  def wrapper(*args, **kwargs): 
    field, op_params = evaluate(*args, **kwargs)
    field._op_params = op_params

    if debug_config["debug_dispatch"]:
      print(f"Dispatching {evaluate.__name__} with for types {evaluate.__annotations__}")

    return field
  
  f = _jaxdf_dispatch(wrapper, precedence=precedence)
  return f

def operator(evaluate: Callable = None, precedence: int = 0):
  r'''Decorator for defining operators using multiple dispatch. The type annotation of the
  `evaluate` function are used to determine the dispatch rules.
  
  Generic inputs must have the type-hint `object`.
  
  Args:
    evaluate (Callable): A function with the signature `evaluate(field, *args, **kwargs, params)`.
      It must return a tuple, with the first element being a field and the second
      element being the default parameters for the operator.
    precedence (int): The precedence of the operator if an ambiguous match is found.
  
  Returns:
    Callable: The operator function with signature `evaluate(field, *args, **kwargs, params)`.
    
  !!! example
      ```python
      from jaxdf import operator
      
      @operator(precedence=1)
      def square_plus_two(x: OnGrid, params=2):
        new_params = (x.params**2) + params
        return x.replace_params(new_params), params
        
      @operator
      def square_plus_two(x: Continuous, params=2):
        get_x = x.aux['get_field']
        def new_get_field(p, coords):
          return get_x(p, coords)**2 + params
        return Continuous(x.params, x.domain, new_get_field), params
      ```
  '''
  if evaluate is None:
    # Returns the decorator
    def decorator(evaluate):
      return _operator(evaluate, precedence)
    return decorator
  else:
    return _operator(evaluate, precedence)

def new_discretization(cls):
  r'''Wrapper around `jax.tree_util.register_pytree_node_class` that can
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
  '''
  return register_pytree_node_class(cls)

@new_discretization
class Field(object):
  r'''The base-class for all discretizations. This class is also responsible for binding the operators in `jaxdf.operators.magic` to
  the magic methods of the discretization.
  
  Normally you should not use this class directly, but instead use the `new_discretization` decorator to register
  a new discretization based on this class.
  '''
  def __init__(self, 
    params,
    domain,
    dims=1,
    aux = None,
  ):
    self.params = params
    self.domain = domain
    self.dims = dims
    self.aux = aux
  
  def tree_flatten(self):
    children = (self.params,)
    aux_data = (self.dims, self.domain, self.aux)
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    params = children[0]
    dims, domain, aux = aux_data
    a = cls(params, dims=dims, domain=domain, aux=aux)
    return a
  
  def __repr__(self):#
    classname = self.__class__.__name__
    return f"Field {classname}"
  
  def __str__(self):
    return self.__repr__()
    
  def replace_params(self, new_params):
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
def __add__(self, other, params=Params):
  raise NotImplementedError(f"Function not implemented for {type(self)} and {type(other)}")

@operator
def __radd__(self, other, params=Params):
  raise NotImplementedError(f"Function not implemented for {type(self)} and {type(other)}")

@operator
def __sub__(self, other, params=Params):
  raise NotImplementedError(f"Function not implemented for {type(self)} and {type(other)}")

@operator
def __rsub__(self, other, params=Params):
  raise NotImplementedError(f"Function not implemented for {type(self)} and {type(other)}")

@operator
def __mul__(self, other, params=Params):
  raise NotImplementedError(f"Function not implemented for {type(self)} and {type(other)}")

@operator
def __rmul__(self, other, params=Params):
  raise NotImplementedError(f"Function not implemented for {type(self)} and {type(other)}")

@operator
def __neg__(self, params=Params):
  raise NotImplementedError(f"Function not implemented for {type(self)}")

@operator
def __pow__(self, other, params=Params):
  raise NotImplementedError(f"Function not implemented for {type(self)} and {type(other)}")

@operator
def __rpow__(self, other, params=Params):
  raise NotImplementedError(f"Function not implemented for {type(self)} and {type(other)}")

@operator
def __truediv__(self, other, params=Params):
  raise NotImplementedError(f"Function not implemented for {type(self)} and {type(other)}")

@operator
def __rtruediv__(self, other, params=Params):
  raise NotImplementedError(f"Function not implemented for {type(self)} and {type(other)}")

  # Lifted jax functions for convenience
def params_map(
  f: Callable, 
  field: Field, 
  *rest
) -> Field:
  r'''Maps a function to the parameters of a Field. 
  
  Since a Field is a pytree, this is equivalent to (and implemented
  using) `jax.tree_util.tree_map`
  
  Returns a field with the same type of `f`, with updated
  parameters
  
  Args:
    f (Callable): A function that is applied to all the leaves of the
      parameters of the input fields
    field (Field): The field to map the function to
    *rest: Optional additional fields to map the function to
    
  Returns:
    Field: A field with the same discretization as `field`, with updated parameters.
  '''
  return tree_map(f, field, *rest)