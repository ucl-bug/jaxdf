import jax
import pytest

from jaxdf.mods import Module


@pytest.fixture
def test_module():

  class TestModule(Module):
    a: float = 1.0
    b: float = 2.0

  return TestModule()


def test_replace_one_attribute_positional(test_module):
  m2 = test_module.replace('a', 3.0)
  assert m2.a == 3.0 and m2.b == 2.0


def test_replace_one_attribute_keyword(test_module):
  m2 = test_module.replace(a=3.0)
  assert m2.a == 3.0 and m2.b == 2.0


def test_replace_multiple_attributes_positional(test_module):
  m2 = test_module.replace('a', 3.0, 'b', 4.0)
  assert m2.a == 3.0 and m2.b == 4.0


def test_replace_multiple_attributes_mixed(test_module):
  m2 = test_module.replace('a', 3.0, b=4.0)
  assert m2.a == 3.0 and m2.b == 4.0


def test_replace_invalid_attribute(test_module):
  with pytest.raises(AttributeError):
    test_module.replace('c', 3.0)


def test_replace_invalid_args_length(test_module):
  with pytest.raises(AssertionError):
    test_module.replace('a', 3.0, 'b')


def test_replace_duplicate_names(test_module):
  with pytest.raises(AssertionError):
    test_module.replace('a', 3.0, 'a', 4.0)


# Tests for jit compatibility
def test_jit_replace_one_attribute_positional(test_module):
  jit_replace = jax.jit(lambda m, value: m.replace('a', value))
  m2 = jit_replace(test_module, 3.0)
  assert m2.a == 3.0 and m2.b == 2.0


def test_jit_replace_one_attribute_keyword(test_module):
  jit_replace = jax.jit(lambda m, a: m.replace(a=a))
  m2 = jit_replace(test_module, 3.0)
  assert m2.a == 3.0 and m2.b == 2.0


def test_jit_replace_multiple_attributes_positional(test_module):
  jit_replace = jax.jit(
      lambda m, value1, value2: m.replace('a', value1, 'b', value2))
  m2 = jit_replace(test_module, 3.0, 4.0)
  assert m2.a == 3.0 and m2.b == 4.0


def test_jit_replace_multiple_attributes_mixed(test_module):
  # For mixed arguments, we need to define a wrapper function
  def replace_wrapper(m, a, b):
    return m.replace('a', a, b=b)

  jit_replace = jax.jit(replace_wrapper)
  m2 = jit_replace(test_module, 3.0, 4.0)
  assert m2.a == 3.0 and m2.b == 4.0
