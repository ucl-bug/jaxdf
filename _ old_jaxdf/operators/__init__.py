from arithmetic import *
from jaxdf.discretization import *
from jaxdf.core import bind_primitive


@bind_primitive
def add(x: Arbitrary, y: Arbitrary) -> Arbitrary:
    # Process parameters
    new_params = [x.params, y.params]

    # Process get_field
    def get_field(p, x):
        return x.discretization.get_field(p, x) + y.discretization.get_field(p, y)

    return None, Arbitrary(
        x.domain, get_field, 
    )