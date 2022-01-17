from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
from jax.tree_util import register_pytree_node
from jax.tree_util import register_pytree_node_class
from jax import jit
import jax
import os
from plum import dispatch
from functools import wraps


def show_example(structured):
    flat, tree = tree_flatten(structured)
    unflattened = tree_unflatten(tree, flat)
    print("structured={}\n  flat={}\n  tree={}\n  unflattened={}".format(
        structured, flat, tree, unflattened))

class _Fourier(object):
    def __init__(self, ext_params={"bias": 4.0}):
        self.params = jnp.ones(1000)
        self.aux = ext_params
        
    def __call__(self, x):
        return self.params + x + self.aux["bias"]

    def __repr__(self):
        return f"Fourier {self.params}"

@register_pytree_node_class
class Fourier(_Fourier):
    def __repr__(self):
        return f"Fourier {self.params}"
    
    def tree_flatten(self):
        children = (self.params,)
        aux_data = self.aux
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        a = cls(aux_data)
        a.params = children[0]
        return a

class Parameter:
    def __init__(self, name, value):
        self.name = name
        self.value = value

def bind_primitive(f):
    dispatched = dispatch(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return dispatched(*args, **kwargs)

    wrapper._tracer = True
    return wrapper

@bind_primitive
def func_with_params(a: Fourier):
    constant = Parameter("constant", 2.3)
    new_params = a.params + constant.value

    # Make new instance of a
    new_a = a.__class__()
    new_a.params = new_params
    new_a.aux = a.aux
    
    return new_a

if __name__ == "__main__": 
    field = Fourier()
    show_example(field)
    
    print("---")

    print(jit(func_with_params)(field))