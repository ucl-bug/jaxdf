import logging

import pytest

from jaxdf.core import Field, debug_config, discretization, logger


def test_debug_dict_deprecation_warning():

  # Test for the deprecation warning when setting the 'debug_dispatch' key
  with pytest.warns(
      DeprecationWarning,
      match=
      "debug_dispatch is deprecated. Set the logger level to DEBUG instead."):
    debug_config['debug_dispatch'] = True

  # Test if the logging level is set to DEBUG when 'debug_dispatch' is True
  assert logger.getEffectiveLevel(
  ) == logging.DEBUG, "Logging level not set to DEBUG when 'debug_dispatch' is True"

  # Test for no warning when setting the 'debug_dispatch' key to False
  with pytest.warns(
      DeprecationWarning,
      match=
      "debug_dispatch is deprecated. Set the logger level to DEBUG instead."):
    debug_config['debug_dispatch'] = False

  # Test if the logging level is set to INFO when 'debug_dispatch' is False
  assert logger.getEffectiveLevel(
  ) == logging.INFO, "Logging level not set to INFO when 'debug_dispatch' is False"

  # Test for ValueError when setting a key other than 'debug_dispatch'
  with pytest.raises(ValueError,
                     match="Only debug_dispatch is supported for now"):
    debug_config['other_key'] = True


def test_discretization_deprecation_warning():

  with pytest.warns(
      DeprecationWarning,
      match=
      "jaxdf.discretization is deprecated since the discretization API has been moved to equinox. You don't need this decorator anymore. It iwll now simply act as a pass-through."
  ):

    @discretization
    class MyField(Field):
      pass
