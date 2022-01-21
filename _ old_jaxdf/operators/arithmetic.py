from jaxdf.discretization import *
from jaxdf.primitives import *
from plum import dispatch


@dispatch
def __add__(x: Arbitrary, y: Arbitrary, lift_params=False):
    return AddField()(x, y)

@dispatch
def __add__(x: Arbitrary, y: float):
    return AddScalar()(x, y)

@dispatch
def __add__(x: float, y: Arbitrary):
    return AddScalar()(y, x)