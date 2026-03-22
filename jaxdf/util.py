from jax.numpy import expand_dims, ndarray


def append_dimension(x: ndarray):
  return expand_dims(x, -1)


def update_dictionary(old: dict, new_entries: dict):
  r"""Update a dictionary with new entries.

    Args:
      old (dict): The dictionary to update
      new_entries (dict): The new entries to add to the dictionary

    Returns:
      dict: The updated dictionary
    """
  for key, val in zip(new_entries.keys(), new_entries.values()):
    old[key] = val
  return old


def get_implemented(f):
  r"""Prints the implemented methods of an operator

    Arguments:
      f (Callable): The operator to get the implemented methods of.

    Returns:
      None

    """

  # TODO: Why there are more instances for the same types?

  print(f.__name__ + ":")
  instances = []
  a = f.methods
  for f_instance in a:
    # Get types
    types = f_instance.signature.types

    # Change each type with its classname
    types = tuple(map(lambda x: x.__name__, types))

    # Append
    instances.append(str(types))

  instances = set(instances)
  for instance in instances:
    print(" ─ " + instance)


def get_implementations(f):
  """Returns the implemented type signatures for an operator.

    Args:
        f: An operator function registered via @operator.

    Returns:
        list[tuple[str, ...]]: List of type signature tuples for each
        implementation.

    Example:
        >>> from jaxdf.operators import gradient
        >>> get_implementations(gradient)
        [('Continuous',), ('FiniteDifferences',), ('FourierSeries',)]
    """
  instances = []
  for f_instance in f.methods:
    types = f_instance.signature.types
    type_names = tuple(t.__name__ for t in types)
    if type_names not in instances:
      instances.append(type_names)
  return sorted(instances)


def has_implementation(f, *types):
  """Check if an operator has an implementation for the given types.

    Args:
        f: An operator function registered via @operator.
        *types: The types to check for.

    Returns:
        bool: True if an implementation exists for the given types.

    Example:
        >>> from jaxdf.operators import gradient
        >>> from jaxdf.discretization import FourierSeries
        >>> has_implementation(gradient, FourierSeries)
        True
    """
  type_names = tuple(t.__name__ for t in types)
  return type_names in get_implementations(f)
