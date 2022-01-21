
import inspect
from functools import partial, wraps
from typing import get_type_hints, Any, NewType, TypeVar

from jax.tree_util import register_pytree_node, register_pytree_node_class, tree_map, tree_multimap

from plum import dispatch


Params = None
@register_pytree_node_class
class Operator:
  def __init__(self, setup_fun, evaluate):
    self.setup_fun = setup_fun
    self.evaluate = evaluate

  def __call__(self, *args, **kwargs):
    # Check if tracer exists in any of the arguments and
    # it is not None.
    for arg in args:
      if hasattr(arg, "_tracer") and arg._tracer is not None:
        return self._traced_call(*args, **kwargs)
    for arg in kwargs.values():
      if hasattr(arg, "_tracer") and arg._tracer is not None:
        return self._traced_call(*args, **kwargs)
    return self._call(*args, **kwargs)

  def _traced_call(self, *args, **kwargs):
    pass

  def _call(self, *args, **kwargs):
    # Construct the operator parameters
    op_params = self.setup_fun(*args, **kwargs)
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


def _operator(evaluate, setup_fun, precedence=0):
  op = Operator(setup_fun, evaluate)
  
  
  @wraps(evaluate)
  def _operator(*args, **kwargs):
    return op(*args, **kwargs)

  return dispatch(_operator, precedence=precedence)

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

'''

def operator(evaluate, precedence=0):
  setup_fun = lambda *args, **kwargs: None
  
  @wraps(evaluate)
  def evaluate_wrapper(*args, params, **kwargs):
    return evaluate(*args, **kwargs)
  
  return _operator(evaluate_wrapper, setup_fun, precedence)

def parametrized_operator(setup_fun, precedence=0):
  def decorate(evaluate):
    op = _operator(evaluate, setup_fun, precedence)
    return op
  return decorate
'''

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
  