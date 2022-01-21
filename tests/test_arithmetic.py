from jaxdf import *
from jax import jit, make_jaxpr
import inspect

# Fields on grid
x = OnGrid(1.0, 'domain')
y = OnGrid(2.0, 'domain')

def test_add():
    z = x + y
    assert z.params == 3.0
    assert type(z) == OnGrid

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
  print(make_jaxpr(add)(x, 1.0))
    
if __name__ == '__main__':
    #test_add()
    #test_jit()
    test_jit_with_float()