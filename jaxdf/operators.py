from typing import Callable
from jaxdf.geometry import Staggered
from jaxdf.core import operator


class Operator(object):
    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            for arg in args:
                if not isinstance(arg, float) and not isinstance(arg, int):
                    return getattr(arg.discretization, self.name)(*args, **kwargs)
        else:
            for arg in kwargs.values():
                if not isinstance(arg, float) and not isinstance(arg, int):
                    return getattr(arg.discretization, self.name)(*args, **kwargs)

        raise RuntimeError(f"Operator {self.name} not found")

    def eval(self, *args, **kwargs):
        # Wrap it with an @operator decorator
        @operator(debug=False)
        def _eval(*args, **kwargs):
            return self(*args, **kwargs)
        
        return _eval(*args, **kwargs)

add = Operator("add")
add_scalar = Operator("add_scalar")
div = Operator("div")
invert = Operator("invert")
mul = Operator("mul")
mul_scalar = Operator("mul_scalar")
div = Operator("div")
div_scalar = Operator("div_scalar")
power = Operator("power")
power_scalar = Operator("power_scalar")
reciprocal = Operator("reciprocal")

gradient = Operator("gradient")
nabla_dot = Operator("nabla_dot")
diag_jacobian = Operator("diag_jacobian")
sum_over_dims = Operator("sum_over_dims")
laplacian = Operator("laplacian")

class OperatorWithArgs(Operator):
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, u):
        return getattr(u.discretization, self.name)(u, *self.args, **self.kwargs)


def staggered_grad(c_ref: float, dt: float, direction: Staggered):
    return OperatorWithArgs("staggered_grad", c_ref=c_ref, dt=dt, direction=direction)


def staggered_diag_jacobian(c_ref: float, dt: float, direction: Staggered):
    return OperatorWithArgs(
        "staggered_diag_jacobian", c_ref=c_ref, dt=dt, direction=direction
    )

def project(u: float, discretization):
    return OperatorWithArgs("project", u=u, discretization=discretization)

class elementwise(Operator):
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, u):
        return u.discretization.elementwise(u, self.func)


class dirichlet(Operator):
    def __init__(self, bc_bvalue):
        self.bc_value = bc_bvalue

    def __call__(self, v):
        return self.u.discretization.dirichlet(self.u, v)
