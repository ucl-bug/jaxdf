"""
This module contains dummy operators for testing purposes.
"""
from jaxdf.core import operator
from jaxdf.discretization import Continuous, Field, OnGrid


@operator  # type: ignore
def dummy(x: Field, *, params=None):
    """Dummy operator for testing purposes."""
    return x


@operator  # type: ignore
def dummy(x: OnGrid, *, params=None):
    r"""A dummy operator that is useful for debugging."""
    if params is None:
        params = {"k": 3.0}
    return params["k"] * x


@operator  # type: ignore
def dummy(x: Continuous, *, params=None):
    if params is None:
        params = {"k": 3.0}
    get_x = x.aux["get_field"]

    def get_fun(p__par, coords):
        p, params = p__par
        return get_x(p, coords) + params["k"]

    return x.update_fun_and_params([x.params, params], get_fun), params


def yummy_init(x: OnGrid, *args, **kwargs):
    return {"k": 3.0}


@operator(init_params=yummy_init)  # type: ignore
def yummy(x: OnGrid, *, params=None):
    r"""A dummy operator that is useful for debugging."""
    return params["k"] * x
