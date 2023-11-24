import inspect
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


def check_eval_init_signatures(evaluate: Callable, init_params: Callable):
  sig_init_params = inspect.signature(init_params)
  sig_evaluate = inspect.signature(evaluate)

  # Extract positional and keyword arguments from both signatures
  evaluate_pos_args = [
      param.name for param in sig_evaluate.parameters.values()
      if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
  ]
  evaluate_kw_args = [
      param.name for param in sig_evaluate.parameters.values()
      if param.kind == inspect.Parameter.KEYWORD_ONLY
  ]

  init_params_var_args = any(param.kind == inspect.Parameter.VAR_POSITIONAL
                             for param in sig_init_params.parameters.values())
  init_params_var_kwargs = any(
      param.kind == inspect.Parameter.VAR_KEYWORD
      for param in sig_init_params.parameters.values())

  init_params_pos_args = [
      param.name for param in sig_init_params.parameters.values()
      if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
  ]
  init_params_kw_args = [
      param.name for param in sig_init_params.parameters.values()
      if param.kind == inspect.Parameter.KEYWORD_ONLY
  ]

  # Remove the 'params' argument from the evaluate function signature
  evaluate_kw_args = [param for param in evaluate_kw_args if param != 'params']

  # Check compatibility for positional arguments
  if not init_params_var_args and len(evaluate_pos_args) > 0:
    # Check that the positional arguments of the evaluate function are
    # present in the init_params function
    for arg in evaluate_pos_args:
      if arg not in init_params_pos_args:
        raise SignatureError(
            f"The init_params function {init_params.__name__} does not have the positional argument '{arg}' of the evaluate function {evaluate.__name__}. You can add it as a positional argument or add *args to the init_params function."
        )

  # Check compatibility for keyword arguments
  if not init_params_var_kwargs and len(evaluate_kw_args) > 0:
    for kwarg in evaluate_kw_args:
      if kwarg not in init_params_kw_args:
        raise SignatureError(
            f"The init_params function {init_params.__name__} must have **kwargs or explicitly define the keyword argument '{kwarg}' of the evaluate function {evaluate.__name__}."
        )


def add_defaults(f, kwargs, skip=[]):
  # Get defaults
  signature = inspect.signature(f)
  defaults = {
      k: v.default
      for k, v in signature.parameters.items()
      if v.default is not inspect.Parameter.empty
  }

  # Add the defaults to the kwargs if they are not already there
  for k, v in defaults.items():
    if k not in kwargs:
      kwargs[k] = v

  # Remove the arguments in skip
  for k in skip:
    if k in kwargs:
      del kwargs[k]

  return kwargs
