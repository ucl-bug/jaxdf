from jaxdf.core import operator, Params
from jaxdf.discretization import *

def setup_dummy(x: OnGrid):
  return {"k": 3}

@operator(setup_fun=setup_dummy)
def dummy(x: OnGrid, params=Params):
  return params["k"]*x

