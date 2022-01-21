
import inspect
from functools import partial, wraps
from typing import get_type_hints, Any, NewType, TypeVar

from jax.tree_util import register_pytree_node, register_pytree_node_class, tree_map, tree_multimap
from numpy import issubdtype

from plum import dispatch

from jaxdf import util

Params = None


@register_pytree_node_class
class Parameters(object):
  def __init__(self, params):
    self.params = params
    
  def tree_flatten(self):
    children = (self.params,)
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(children[0])
  
@register_pytree_node_class
class Operator:
  r'''Operator class used to define an operator. It 
  takes care of dealing with traced executions when
  `@collect_params ` is invoked.
  
  Operators are construced with the `@operator` decorator
  '''
  def __init__(self, setup_fun, evaluate):
    self.setup_fun = setup_fun
    self.evaluate = evaluate

  def __call__(self, *args, **kwargs):
    if 'params' in kwargs.keys():
      if isinstance( kwargs['params'], Parameters):
        ext_params = kwargs['params']
        del kwargs['params']
        return self._parametrized_call(ext_params, *args, **kwargs)
      else:
        raise ValueError('The `params` input variable is restricted to Parameters')
    
    # Otherwise the standard call
    return self._call(*args, **kwargs)

  def _call(self, *args, **kwargs):
    # Construct the operator parameters
    op_params = self.setup_fun(*args, **kwargs)
    # Runs the parametrized function
    field = self.evaluate(*args, **kwargs, params = op_params)
    return field
  
  def _parametrized_call(
    self, 
    ext_params: Parameters, 
    *args, **kwargs
  ):
    # Construct the operator parameters
    op_params = self.setup_fun(*args, **kwargs)
    
    # Update it with the provided keys
    op_params = util.update_dictionary(op_params, ext_params.params)
    
    # Runs the parametrized function
    field = self.evaluate(*args, **kwargs, params = op_params)
    return field

  def tree_flatten(self):
    children = {}
    aux_data = (self.setup_fun, self.evaluate)
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(setup_fun=aux_data[0], evaluate=aux_data[1])
  
@register_pytree_node_class
class ConcreteOperator(Operator):
  def __init__(self, setup_fun, evaluate):
    self.setup_fun = setup_fun
    self.evaluate = evaluate
    self.params = []
  
  def tree_flatten(self):
    children = (self.params,)
    aux_data = (self.setup_fun, self.evaluate)
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    a = cls(setup_fun=aux_data[0], evaluate=aux_data[1])
    a.params = children[0]
    return a

def _operator(evaluate, setup_fun, precedence=0):
  op = Operator(setup_fun, evaluate)
  
  @wraps(evaluate)
  def wrapper(*args, **kwargs):
    return op(*args, **kwargs)

  f = dispatch(wrapper, precedence=precedence)
  
  # Add get_params function
  def get_params(*args, **kwargs):
    return Parameters(setup_fun(*args, **kwargs))
  f.get_params = get_params
  
  return f

def operator(evaluate=None, setup_fun=None, precedence=0):
  if setup_fun is None:
    setup_fun = lambda *args, **kwargs: None
    
  if evaluate is None:
    # Returns the decorator
    def decorator(evaluate):
      return _operator(evaluate, setup_fun, precedence)
    return decorator
  else:
    return _operator(evaluate, setup_fun, precedence)

# Lifted jax functions for convenience
def params_map(f, field, *rest):
  r'''Maps a function to the parameters of a Field. 
  
  Since a Field is a pytree, this is quivalent to (and implemented
  using) `jax.tree_util.tree_map`
  
  Returns a field with the same time of `field`, with updated
  parameters
  '''
  if len(rest) > 0:
    for a in rest:
      assert type(a) == type(field)
  return tree_map(f, field, *rest)


class Field(object):
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
    
  def __repr__(self):#
    classname = self.__class__.__name__
    return f"Field {classname}\n - Params:{self.params}\n"
  
  def __str__(self):
    return self.__repr__()
    
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
  
  def replace_params(self, new_params):
    return self.__class__(new_params, self.domain, self.dims, self.aux)
  
  # Dummy magic functions to make it work with
  # the dispatch method
  def __add__(self, other):
    @operator
    def __add__(x, y):
      raise NotImplementedError(
        f"Unimplemented __add__ for types {type(self)} {type(other)}"
      )
    return __add__(self, other)
  
  def __bool__(self):
    @operator
    def __bool__(x):
      raise NotImplementedError(
        f"Unimplemented __bool__ for type {type(self)}"
      )
    return __bool__(self)
 
  def __divmod__(self, other):
    @operator
    def __divmod__(x, y):
      raise NotImplementedError(f"Unimplemented __divmod__ for types {type(self)} {type(other)}")
    return __divmod__(self, other)    
  
  def __float__(self):
    @operator
    def __float__(x):
      raise NotImplementedError(f"Unimplemented __float__ for type {type(self)}")
    return __float__(self)

  def __mul__(self, other):
    @operator
    def __mul__(x, y):
      return None
    return __mul__(self, other)
  
  def __neg__(self):
    @operator
    def __neg__(x):
      return None
    return __neg__(self)
  
  def __pow__(self, other):
    @operator
    def __pow__(x, y):
      raise NotImplementedError(f"Unimplemented __pow__ for types {type(self)} {type(other)}")
    return __pow__(self, other)
      
  def __radd__(self, other):
    return self.__add__(other)
  
  def __rmul__(self, other):
    @operator
    def __mul__(x, y):
      return None
    return __mul__(self, other)
  
  def __rpow__(self, other):
    @operator
    def __rpow__(x, y):
      return None
    return __rpow__(self, other)
  
  def __rsub__(self, other):
    @operator
    def __rsub__(x, y):
      return None
    return __rsub__(self, other)
  
  def __sub__(self, other):
    @operator
    def __sub__(x, y):
      return None
    return __sub__(self, other)