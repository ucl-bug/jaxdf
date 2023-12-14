import equinox as eqx


class Module(eqx.Module):
  """
    A custom module inheriting from Equinox's Module class.
    """

  def replace(self, *args, **kwargs):
    """
        Replaces the attribute of the module with the given name with a new value.

        This method utilizes `eqx.tree_at` to update the attribute in a functional
        manner, ensuring compatibility with JAX's functional approach and autodiff capabilities.

        Args:
            name (str): The name of the attribute to be replaced.
            value (PyTree): The new value to set for the attribute. This should be
                            compatible with JAX's PyTree structure.

        Returns:
            A new instance of Module with the specified attribute updated.
            The rest of the module's attributes remain unchanged.

        !!! example
        ```python
            >>> module = jaxdf.Module(weight=1.0, bias=2.0)
            >>> new_module = module.replace(weight=3.0) # Alternatively, module.replace('weight', 3.0)
            >>> new_module.weight == 3.0    # True
        ```
        """
    # Make sure that the number of args is even
    assert len(
        args
    ) % 2 == 0, "The number of arguments must be even, since they are passed as name-value pairs. E.g. `.replace('weight', 1.0, 'bias', 2.0)`"

    # Check that no name is repeated
    names = args[::2] + tuple(kwargs.keys())
    if len(args) > 0:
      duplicated_names = [name for name in names if names.count(name) > 1]
      assert len(
          duplicated_names
      ) == 0, f"The following names are repeated: {duplicated_names}"

    values = args[1::2] + tuple(kwargs.values())

    f = lambda m: [m.__getattribute__(name) for name in names]
    return eqx.tree_at(f, self, values)
