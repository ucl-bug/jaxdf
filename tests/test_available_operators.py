import pytest
from plum.resolver import NotFoundLookupError

from jaxdf.operators import (derivative, diag_jacobian, dot_product, dummy,
                             get_component, gradient, heterog_laplacian,
                             laplacian, shift_operator, sum_over_dims, yummy)

unary_operators = [
    derivative,
    diag_jacobian,
    dummy,
    get_component,
    gradient,
    heterog_laplacian,
    laplacian,
    shift_operator,
    sum_over_dims,
    yummy,
]

binary_operators = [dot_product]


@pytest.mark.parametrize("operator", unary_operators)
def test_abstract_unary_operator(operator):
  with pytest.raises(NotFoundLookupError):
    operator(None)


@pytest.mark.parametrize("operator", binary_operators)
def test_abstract_binary_operator(operator):
  with pytest.raises(NotFoundLookupError):
    operator(None, None)


if __name__ == "__main__":
  compose(None)
