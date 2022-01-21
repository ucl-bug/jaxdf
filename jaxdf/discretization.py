from typing import Callable
from jax.random import PRNGKey
from jax import eval_shape
from jax.tree_util import register_pytree_node, register_pytree_node_class
from jax import tree_util
from jaxdf.core import operator, Params, Field

new_discretization = register_pytree_node_class

@new_discretization
class Linear(Field):
  def __init__(
    self,
    params,
    domain,
    dims=1,
    aux = None,
    
  ):
    super().__init__(params, domain, dims, aux)
               

@new_discretization
class Continuous(Field):
  def __init__(
    self,
    params,
    domain,
    dims=1,
    aux = None,
    
  ):
    super().__init__(params, domain, dims, aux)
  
  def get_field(self, params, x):
    return self.aux["get_field"](params, x)
  
  @classmethod
  def from_function(
    cls, 
    domain,
    init_fun: Callable, 
    get_field: Callable,
    seed
  ):
    params = init_fun(seed)
    x = domain.origin
    dims = eval_shape(get_field, params, x).shape
    return cls(
      params,
      domain=domain,
      dims=dims,
      aux={"get_field": get_field},
      
    )
  
  @classmethod
  def from_fun_and_params(
    cls,
    params,
    domain,
    get_field
  ):
    dims = eval_shape(get_field, params, domain.origin).shape[-1]
    return cls(
      params,
      domain=domain,
      dims=dims,
      aux={"get_field": get_field},
    )
      
@new_discretization
class OnGrid(Linear):
  def __init__(
      self,
      params,
      domain,
      dims=1,
      aux = None,
      
  ):
    super().__init__(params, domain, dims, aux)
  
  @classmethod
  def from_grid(cls, grid_values, domain):
    return cls(grid_values, domain, grid_values.shape[-1])
  
@new_discretization
class FiniteDifferences(OnGrid):
  def get_field_on_grid(self):
    return self.params