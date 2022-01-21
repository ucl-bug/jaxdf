from jaxdf import *
from jax import jit, make_jaxpr
import inspect
import numpy as np

ATOL=1e-6

# Fields on grid
x = OnGrid(1.0, 'domain')
y = OnGrid(2.0, 'domain')

def test_add():
  z = x + y
  assert z.params == 3.0
  assert type(z) == OnGrid
  
  z = x + 5.
  assert z.params == 6.0

def test_sub():
  z = x - y
  assert z.params == -1.0
  assert type(z) == OnGrid
  
  z = x - 2.0
  assert z.params == -1.0

def test_jit():
  @jit
  def prod(x, y):
    return x * y
  return prod(x, y)

def test_jit_with_float():
  @jit
  def add(x, y):
    return x + y * 10
  
  _ = add(x,y)
  _ = add(x, 6.0)
  _ = add(-5.0, x)
    
if __name__ == '__main__':
    test_add()
    test_sub()
    test_jit()
    test_jit_with_float()