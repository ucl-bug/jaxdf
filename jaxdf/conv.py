from typing import List, Tuple, Union, no_type_check

import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp


def reflection_conv(kernel: jnp.ndarray,
                    array: jnp.ndarray,
                    reverse: bool = True) -> jnp.ndarray:
  r"""Convolves an array with a kernel, using reflection padding.
    The kernel is supposed to have the same number of dimensions as the array.

    Args:
      kernel (jnp.ndarray): The kernel to convolve with.
      array (jnp.ndarray): The array to convolve.
      reverse (bool, optional): Whether to reverse the kernel before convolving.
        Defaults to True.

    Returns:
      The convolved array.
    """
  # Reflection padding the array appropriately
  pad = [((x - 1) // 2, (x - 1) // 2) for x in kernel.shape]
  f = jnp.pad(array, pad, mode="wrap")

  if reverse:
    # Reverse the kernel over all axes
    kernel = jnp.flip(kernel, axis=tuple(range(kernel.ndim)))

  # Apply kernel
  return jsp.signal.convolve(f, kernel, mode="valid")


def bubble_sort_abs_value(
    points_list: List[Union[float, int]]) -> List[Union[float, int]]:
  r"""Sorts a sequence of grid points by their absolute value.

    Sorting is done __in place__. This function is written with numpy, so it can't
    be transformed by JAX.

    !!! example
        ```python
        lst = [-3, -2, -1, 0, 1, 2, 3]
        sorted_lst = bubble_sort_abs_value(lst)
        print(sorted_lst)
        # [0, 1, -1, 2, -2, 3, -3]
        ```

    Args:
      points_list (List[Union[float, int]]): The grid points to sort.

    Returns:
      The sorted grid points.
    """

  for i in range(len(points_list)):
    for j in range(0, len(points_list) - i - 1):
      magnitude_condition = abs(points_list[j]) > abs(points_list[j + 1])
      same_mag_condition = abs(points_list[j]) == abs(points_list[j + 1])
      sign_condition = np.sign(points_list[j]) < np.sign(points_list[j + 1])
      if magnitude_condition or (same_mag_condition and sign_condition):
        temp = points_list[j]
        points_list[j] = points_list[j + 1]
        points_list[j + 1] = temp

  return points_list


# TODO (astanziola): This fails on mypy for some reason, but can't work out how to fix.
@no_type_check
def fd_coefficients_fornberg(
    order: int, grid_points: List[Union[float, int]],
    x0: Union[float, int]) -> Tuple[List[None], List[Union[float, int]]]:
  r"""Generate finite difference stencils for a given order and grid points, using
    the Fornberg algorithm described in [[Fornberg, 2018]](https://web.njit.edu/~jiang/math712/fornberg.pdf).

    The grid points can be placed in any order, can be at arbitrary locations (for example, to implemente staggered
    stencils) and don't need to be equidistant.
    The stencil is evaluated for a point in `x0`. Note that setting `order=0` will generate interpolation coefficients
    for the point `x0`.

    !!! example
        ```python
        grid_points = [0, 1, 2, -1, -2]
        x0 = 0.0
        order = 2
        stencil, nodes = fd_coefficients_fornberg(order, grid_points, x0)
        print(f"Stencil: {stencil}, Nodes: {nodes}")
        # Stencil: [-0.08333333  1.33333333 -2.5         1.33333333 -0.08333333], Nodes: [-2 -1  0  1  2]
        ```

    Args:
      order (int): The order of the stencil.
      grid_points (List[Union[float, int]]): The grid points to use.
      x0 (Union[float, int]): The point at which to evaluate the stencil.

    Returns:
      The stencil and the grid points where the stencil is evaluated.
    """
  # from Generation of Finite Difference Formulas on Arbitrarily Spaced Grids
  # Bengt Fornberg, 1998
  # https://web.njit.edu/~jiang/math712/fornberg.pdf
  M = order
  N = len(grid_points) - 1

  # Sort the grid points
  alpha = bubble_sort_abs_value(grid_points)
  delta = dict()    # key: (m,n,v)
  delta[(0, 0, 0)] = 1.0
  c1 = 1.0

  for n in range(1, N + 1):
    c2 = 1.0
    for v in range(n):
      c3 = alpha[n] - alpha[v]
      c2 = c2 * c3
      if n < M:
        delta[(n, n - 1, v)] = 0.0
      for m in range(min([n, M]) + 1):
        d1 = delta[(m, n - 1, v)] if (m, n - 1, v) in delta.keys() else 0.0
        d2 = (delta[(m - 1, n - 1, v)] if
              (m - 1, n - 1, v) in delta.keys() else 0.0)
        delta[(m, n, v)] = ((alpha[n] - x0) * d1 - m * d2) / c3

    for m in range(min([n, M]) + 1):
      d1 = (delta[(m - 1, n - 1, n - 1)] if
            (m - 1, n - 1, n - 1) in delta.keys() else 0.0)
      d2 = delta[(m, n - 1, n - 1)] if (m, n - 1,
                                        n - 1) in delta.keys() else 0.0
      delta[(m, n, n)] = (c1 / c2) * (m * d1 - (alpha[n - 1] - x0) * d2)
    c1 = c2

  # Extract the delta with m = M and n = N
  coeffs = [None] * (N + 1)
  for key in delta:
    if key[0] == M and key[1] == N:
      coeffs[key[2]] = delta[key]

  # sort coefficeient and alpha by alpha
  idx = np.argsort(alpha)
  alpha = np.take_along_axis(np.asarray(alpha), idx, axis=-1)
  coeffs = np.take_along_axis(np.asarray(coeffs), idx, axis=-1)

  return coeffs, alpha
