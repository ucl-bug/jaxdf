import numpy as np
from jax import numpy as jnp
from pytest import mark

from jaxdf import conv


def test_reflection_conv():
    array = jnp.zeros((5, 6))
    array = array.at[2, 3].set(1.0)
    kernel = jnp.ones((2, 3))

    output = conv.reflection_conv(kernel, array)
    true_output = jnp.array(
        [
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ]
    )
    assert jnp.allclose(output, true_output)


@mark.parametrize(
    "pair",
    [
        {"in_list": [-3, -2, -1, 0, 1, 2, 3], "out_list": [0, 1, -1, 2, -2, 3, -3]},
        {
            "in_list": [0.5, 1.5, -0.5, 2.5, -1.5, -2.5],
            "out_list": [0.5, -0.5, 1.5, -1.5, 2.5, -2.5],
        },
    ],
)
def test_bubble_sort_abs(pair):
    in_list, out_list = pair.values()
    assert conv.bubble_sort_abs_value(in_list) == out_list


@mark.parametrize(
    "pair",
    [
        {"points": [-1, 0, 1], "order": 1, "stencil": [-0.5, 0, 0.5]},
        {
            "points": [-2, -1, 0, 1, 2],
            "order": 1,
            "stencil": [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0],
        },
        {
            "points": [-2, -1, 0, 1, 2],
            "order": 2,
            "stencil": [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0],
        },
        {"points": [-0.5, 0.5], "order": 0, "stencil": [0.5, 0.5]},
        {
            "points": [-0.5, 0.5, 1.5],
            "order": 0,
            "stencil": [3.0 / 8.0, 3.0 / 4.0, -1.0 / 8.0],
        },
        {
            "points": [-0.5, 0.5, 1.5, 2.5],
            "order": 1,
            "stencil": [-23.0 / 24, 7.0 / 8.0, 1.0 / 8.0, -1.0 / 24.0],
        },
    ],
)
def test_fornberg_coefficients(pair):
    points, order, stencil_true = pair.values()
    stencil, grid_points = conv.fd_coefficients_fornberg(order, points, 0.0)
    assert np.allclose(stencil, stencil_true)
