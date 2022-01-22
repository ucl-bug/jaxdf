from jaxdf.core import operator, Params
from jaxdf.discretization import FiniteDifferences, OnGrid
from jax import jit
import jax
from plum import dispatch

class Number(object):
  def __init__(self, value, aux):
    self.value = value
    self.aux= aux
    
  def __repr__(self):
    return f"Number {self.value}"
  
  def __str__(self):
    return self.__repr__()
    
def tree_flatten(v):
  children = (v.value,)
  aux_data = (v.aux,)
  return (children, aux_data)

def tree_unflatten(aux_data, children):
  value = children[0]
  aux = aux_data[0]
  return Number(value, aux)

jax.tree_util.register_pytree_node(Number, tree_flatten, tree_unflatten)

class Fraction(object):
  def __init__(self, numerator, denominator, aux):
    self.numerator = numerator
    self.denominator = denominator
    self.aux = aux
    
  def __repr__(self) -> str:
      return f"Fraction({self.numerator}, {self.denominator})"
    
def tree_flatten(v):
  children = (v.numerator, v.denominator)
  aux_data = (v.aux,)
  return (children, aux_data)

def tree_unflatten(aux_data, children):
  numerator = children[0]
  denominator = children[1]
  aux = aux_data[0]
  return Fraction(numerator, denominator, aux)

jax.tree_util.register_pytree_node(Fraction, tree_flatten, tree_unflatten)

@dispatch
def f(x: Number):
  return Number(x.value+1, x.aux)

@dispatch
def f(x: Fraction):
  return Fraction(x.numerator-1, x.denominator+1, x.aux)

if __name__ == "__main__":
  with jax.checking_leaks():
    x = Number(1, {"a": 1})
    y = Fraction(2, 3, {"b": 2})
    z = f(x)
    print(z)
    z = f(y)
    print(z)
    q = f(z)
    print(q)