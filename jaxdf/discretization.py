from typing import Callable
from jax.random import PRNGKey
from jax import eval_shape
from jax.tree_util import register_pytree_node, register_pytree_node_class
from jax import tree_util
from jaxdf.core import operator, Params

new_discretization = register_pytree_node_class


@new_discretization
class Field(object):
  def __init__(self, 
    params,
    domain,
    dims=1,
    aux = None,
    tracer=None
  ):
    self.params = params
    self.domain = domain
    self.dims = dims
    self.aux = aux
    self._tracer = tracer
    
  def __repr__(self):#
    classname = self.__class__.__name__
    return f"Field {classname}\n - Params:{self.params}"
  
  def __str__(self):
    return self.__repr__()
    
  def tree_flatten(self):
    children = (self.params,)
    aux_data = (self.dims, self.domain, self.aux, self._tracer)
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    params = children[0]
    dims, domain, aux, tracer = aux_data
    a = cls(params, dims=dims, domain=domain, aux=aux, tracer=tracer)
    return a
  
  def replace_params(self, new_params):
    return self.__class__(
        new_params, self.domain, self.dims, self.aux, self._tracer)
  
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
        f"Unimplemented __bool__ for types {type(self)} {type(other)}"
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
      raise NotImplementedError(f"Unimplemented __float__ for types {type(self)} {type(other)}")
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
    
@new_discretization
class Linear(Field):
  def __init__(
    self,
    params,
    domain,
    dims=1,
    aux = None,
    tracer=None
  ):
    super().__init__(params, domain, dims, aux, tracer)
               

@new_discretization
class Continuous(Field):
  def __init__(
    self,
    params,
    domain,
    dims=1,
    aux = None,
    tracer=None
  ):
    super().__init__(params, domain, dims, aux, tracer)
  
  def get_field(self, x):
    return self.aux["get_field"](self.params, x)
  
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
      tracer=None
    )
      
@new_discretization
class OnGrid(Linear):
  def __init__(
      self,
      params,
      domain,
      dims=1,
      aux = None,
      tracer=None
  ):
    super().__init__(params, domain, dims, aux, tracer)
  
@new_discretization
class FiniteDifferences(OnGrid):
  def get_field_on_grid(self):
    return self.params