import jax
from jax import grad, jit, make_jaxpr
from jax import numpy as jnp
from plum.function import NotFoundLookupError
from pytest import mark, raises

from jaxdf import *
from jaxdf.exceptions import SignatureError
from jaxdf.operators.dummy import dummy, yummy

domain = geometry.Domain()

# Abstract fields
ABS_PARAMS = None
ABS_FIELD = Field(ABS_PARAMS, domain)

# Fields on grid
X_PARAMS = jnp.asarray([1.0])
Y_PARAMS = jnp.asarray([2.0])
X = OnGrid(X_PARAMS, domain)
Y = OnGrid(Y_PARAMS, domain)


# Continuous fields
def CONTINUOUS_FUN(p, x):
    return jnp.expand_dims(jnp.sum(p * (x**2)), -1)


A_PARAMS = 5.0
B_PARAMS = 6.0
A = Continuous(A_PARAMS, domain, CONTINUOUS_FUN)
B = Continuous(B_PARAMS, domain, CONTINUOUS_FUN)


@mark.parametrize("field", [ABS_FIELD, X, A])
def test_keyword_parameters(field):
    # Checks that the parameters of an operator can be passed as
    # keyword arguments.
    params = dummy.default_params(field)
    z = dummy(X, params=params)


@mark.parametrize("field", [ABS_FIELD, X, A])
def test_non_keyword_parameters(field):
    # It should not be possible to pass the parameters as
    # positional arguments.
    params = dummy.default_params(field)
    with raises(NotFoundLookupError):
        _ = dummy(X, params)


@mark.parametrize("field", [ABS_FIELD, X, A])
def test_paramfun(field):
    # Tests the dummy operator
    _ = operators.dummy(field)


@mark.parametrize("field", [ABS_FIELD, X, A])
def test_jit_paramfun(field):
    # Tests that the dummy operator can be jitted.
    @jit
    def f(x):
        return operators.dummy(x)

    _ = f(field)


@mark.parametrize(
    "field",
    [
        X,
    ],
)
def test_field_with_init_param_function(field):
    # This function tests an operator where the parameters are
    # initialized using the oporator's init_params function.
    # That is `operator(init_params=init_fun)(operator_function)`.
    _ = yummy(field)


def test_unparametrized_operator():
    # Tests that an operator cannot be defined without parameters.
    with raises(SignatureError):

        @operator
        def f(x: Field):
            return x


def test_operators_with_non_keyword_params():
    # Tests that an operator cannot be defined with non-keyword
    # parameters.
    with raises(SignatureError):

        @operator
        def f(x: Field, params):
            return x


def test_operator_with_constant_params():
    # Tests that an operator can be defined with constant parameters.

    @operator(init_params=constants({"a": 1, "b": 2.0}))
    def f(x: Field, *, params):
        return params["a"] + params["b"]

    z = f(X)
    assert z == 3.0
    default_params = f.default_params(X)
    assert default_params == {"a": 1, "b": 2.0}


@mark.parametrize("field", [ABS_FIELD, X, A])
def test_get_params(field):
    # Tests that the user can get the default parameters of an operator.
    op_params = operators.dummy.default_params(field)


def test_use_params_in_function():
    # Tests that the user can use the parameters of an operator
    # in a function.
    op_params = operators.dummy.default_params(X)

    def f(x, op_params):
        return operators.dummy(x, params=op_params)

    z = f(X, op_params)
    assert z.params == 3.0

    z = jit(f)(X, op_params)
    assert z.params == 3.0

    op_params = operators.dummy.default_params(X)
    z = jit(f)(A, op_params)
    _ = z

    def f(x, coord, op_params):
        b = operators.dummy(x, params=op_params)
        return b(coord)

    z = jit(f)(A, 1.0, op_params)
    _ = make_jaxpr(f)(A, 1.0, op_params)
    _ = z


def test_grad():
    # Checks that `jax.grad` can be applied to functions
    # that use operators.
    def loss(x, y):
        z = x**2 + y * 5 + x * y
        return jnp.sum(z.params)

    gradfn = grad(loss, argnums=(0, 1))
    x_grad, y_grad = gradfn(X, Y)
    _ = x_grad
    assert x_grad.params == 4.0
    assert y_grad.params == 6.0


@mark.parametrize("field", [ABS_FIELD, X, A])
def test_tracer_leaks(field):
    # Checks that the parameters of an operator don't generate
    # tracer leaks when combined with jax transformations
    # such as `jit` and `grad`.
    with jax.checking_leaks():
        test_paramfun(field=field)
        test_jit_paramfun(field=field)
        test_get_params(field=field)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
