from jaxdf import *
from jax import jit, make_jaxpr
import inspect

# Fields on grid
x = OnGrid(1.0, 'domain')
y = OnGrid(2.0, 'domain')

def test_paramfun():
  a = operators.dummy.dummy(x)
  print(a)
    
if __name__ == '__main__':
  test_paramfun()