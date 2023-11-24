import pytest

from jaxdf.signatures import SignatureError, check_eval_init_signatures


def test_compatible_signatures():

  def evaluate(a, b, c=None):
    pass

  def init_params(a, b, c=None):
    pass

  # Should not raise an exception
  check_eval_init_signatures(evaluate, init_params)


def test_incompatible_positional_args():

  def evaluate(a, b, c):
    pass

  def init_params(a, b):
    pass

  with pytest.raises(SignatureError):
    check_eval_init_signatures(evaluate, init_params)


def test_incompatible_keyword_args():

  def evaluate(a, b, *, c):
    pass

  def init_params(a, b):
    pass

  with pytest.raises(SignatureError):
    check_eval_init_signatures(evaluate, init_params)


def test_variable_positional_args():

  def evaluate(a, b):
    pass

  def init_params(*args):
    pass

  # Should not raise an exception
  check_eval_init_signatures(evaluate, init_params)


def test_variable_keyword_args():

  def evaluate(a, b, *, c):
    pass

  def init_params(**kwargs):
    pass

  # Should raise an exception
  with pytest.raises(SignatureError):
    check_eval_init_signatures(evaluate, init_params)


def test_exclude_params_keyword():

  def evaluate(a, *, params, c):
    pass

  def init_params(a, *, c):
    pass

  # Should not raise an exception
  check_eval_init_signatures(evaluate, init_params)


def test_no_arguments():

  def evaluate():
    pass

  def init_params():
    pass

  # Should not raise an exception
  check_eval_init_signatures(evaluate, init_params)


def test_mixed_keyword_and_positional_args():

  def evaluate(a, *, b, c=None):
    pass

  def init_params(a, b, c=None):
    pass

  # Should raise an exception
  with pytest.raises(SignatureError):
    check_eval_init_signatures(evaluate, init_params)


def test_default_values_in_evaluate():

  def evaluate(a, b=2, c=3):
    pass

  def init_params(a, b, c):
    pass

  # Should not raise an exception
  check_eval_init_signatures(evaluate, init_params)


def test_mismatched_default_values():

  def evaluate(a, b=2, c=3):
    pass

  def init_params(a, b=3, c=4):
    pass

  # Should not raise an exception
  check_eval_init_signatures(evaluate, init_params)


def test_only_variable_arguments_in_evaluate():

  def evaluate(*args, **kwargs):
    pass

  def init_params(a, b, c):
    pass

  # Should not raise an exception
  check_eval_init_signatures(evaluate, init_params)


def test_function_annotations():

  def evaluate(a: int, b: str, c: float = 1.0):
    pass

  def init_params(a: int, b: str, c: float):
    pass

  # Should not raise an exception
  check_eval_init_signatures(evaluate, init_params)


def test_no_overlapping_arguments():

  def evaluate(a, b):
    pass

  def init_params(c, d):
    pass

  with pytest.raises(SignatureError):
    check_eval_init_signatures(evaluate, init_params)


def test_functions_with_only_variable_args():

  def evaluate(*args, **kwargs):
    pass

  def init_params(*args, **kwargs):
    pass

  # Should not raise an exception
  check_eval_init_signatures(evaluate, init_params)
