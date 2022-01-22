
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
  
def _operator(evaluate, setup_fun, precedence=0):
   
  @wraps(evaluate)
  def wrapper(*args, **kwargs): 
    
    if 'params' in kwargs.keys():
      ext_params = kwargs['params']
      del kwargs['params']
      
      op_params = setup_fun(*args, **kwargs)
      
      op_params = util.update_dictionary(op_params, ext_params)
      
      # Runs the parametrized function
      field = evaluate(*args, **kwargs, params = op_params)
      return field
    else:
      op_params = setup_fun(*args, **kwargs)
      return evaluate(*args, **kwargs, params = op_params)
      
  f = dispatch(wrapper, precedence=precedence)
  
  # Attach the parameter setup function
  f.get_params = setup_fun
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

def new_discretization(cls):
  
  def tree_flatten(v):
    children = (v.params,)
    aux_data = (v.dims, v.domain, v.aux)
    return (children, aux_data)

  def tree_unflatten(aux_data, children):
    params = children[0]
    dims, domain, aux = aux_data
    a = cls(params, dims=dims, domain=domain, aux=aux)
    return a
  
  register_pytree_node(cls, tree_flatten, tree_unflatten)
  return cls

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