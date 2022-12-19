from inspect import getfullargspec
from typing import Callable


class SignatureError(Exception):
    """Exception raised when the signature of a function does not match the
    expected signature.
    """

    pass


def check_fun_has_params(f: Callable):
    r"""Checks that the function `f` expects the argument `params` as
    a mandatory keyword argument

    Args:
      f (Callable): The function to check.

    Raises:
      SignatureError: If the function does not have the argument `params`
    """
    fun_spec = getfullargspec(f)
    if "params" in fun_spec.args:
        raise SignatureError(
            f"The argument 'params' must be a keyword argument in {f}, not a positional argument. Example: def evaluate(x, *, params): ..."
        )
    if "params" not in fun_spec.kwonlyargs:
        raise SignatureError(
            f"The argument 'params' must be a keyword argument in {f}. Example: def evaluate(x, *, params): ..."
        )
