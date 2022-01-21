from jaxdf import *
from jax import jit, make_jaxpr
import inspect

# Fields on grid
x = OnGrid(1.0, 'domain')
y = OnGrid(2.0, 'domain')

def test_paramfun():
  a = operators.dummy(x)
  
def test_jit_paramfun():
  @jit
  def f(x):
    return operators.dummy(x)
  _ = f(x)

def test_get_params():
  op_params = operators.dummy.get_params(x)
  assert op_params.params['k'] == 3
  
  def f(x, op_params):
    return operators.dummy(x, params=op_params)
  
  z = f(x, op_params)
  assert z.params == 3.0
  
  z = jit(f)(x, op_params)
  assert z.params == 3.0

if __name__ == '__main__':
  test_paramfun()
  test_jit_paramfun()
  test_get_params()