from jaxdf.core import operator, Params
from jaxdf.discretization import *

def _setup_dummy(x: OnGrid):
  return {"k": 3}

def dummy(x: OnGrid, params=Params):
  return params["k"]*x
