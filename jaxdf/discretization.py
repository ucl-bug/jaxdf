from typing import Callable
from jax.random import PRNGKey
from jax import eval_shape, vmap
from jax.tree_util import register_pytree_node, register_pytree_node_class
from jax import tree_util
from jaxdf.core import operator, Params, Field, new_discretization

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
               

@register_pytree_node_class
class Continuous(Field):
  def __init__(
    self,
    params,
    domain,
    get_fun=lambda p,x: None 
  ):
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
    return self.__class__(new_params, self.domain, self.aux["get_field"])
  
  def update_fun_and_params(
    self,
    params,
    get_field
  ):
    return self.__class__(params, self.domain, get_field)
  
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
      aux={"get_field": get_field}
    )
    
  def get_field(self, x):
    return self.aux["get_field"](self.params, x)
  
  def get_field_on_grid(self):
    """V-maps the get_field function over a grid of values"""
    fun = self.aux["get_field"]
    ndims = len(self.domain.N)
    for _ in range(ndims):
        fun = vmap(fun, in_axes=(None, 0))
        
    return fun(self.params, self.domain.grid)
      
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
  
  def get_field_on_grid(self):
    return self.params
  
  