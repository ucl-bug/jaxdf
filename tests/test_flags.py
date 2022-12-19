def test_debug_dispatch():
    from jax import numpy as jnp

    import jaxdf
    from jaxdf.discretization import FourierSeries

    jaxdf.debug_config["debug_dispatch"] = True

    # Generate a new operator
    @jaxdf.operator
    def op(x: FourierSeries, *, params=None):
        return x

    domain = jaxdf.geometry.Domain((1,), (1.0,))
    x = jaxdf.FourierSeries(jnp.asarray([1.0]), domain)

    # Capture the stdout
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        _ = op(x)

    # Check that the output is correct
    output = f.getvalue()[:-1]
    assert (
        output
        == "Dispatching op with for types {'x': <class 'jaxdf.discretization.FourierSeries'>}"
    )


if __name__ == "__main__":
    test_debug_dispatch()
