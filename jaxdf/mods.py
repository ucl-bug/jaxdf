import equinox as eqx
from jaxtyping import PyTree


class JaxDFModule(eqx.Module):
    """
    A custom module inheriting from Equinox's Module class.

    This module is designed to work with JAX and Equinox libraries, providing
    functionalities that are specific to deep learning models and operations in JAX.
    """

    def replace(self, name: str, value: PyTree):
        """
        Replaces the attribute of the module with the given name with a new value.

        This method utilizes `eqx.tree_at` to update the attribute in a functional
        manner, ensuring compatibility with JAX's functional approach and autodiff capabilities.

        Args:
            name (str): The name of the attribute to be replaced.
            value (PyTree): The new value to set for the attribute. This should be
                            compatible with JAX's PyTree structure.

        Returns:
            A new instance of JaxDFModule with the specified attribute updated.
            The rest of the module's attributes remain unchanged.

        Example:
            >>> module = JaxDFModule(...)
            >>> new_module = module.replace("weight", new_weight_value)
        """
        f = lambda m: m.__getattribute__(name)
        return eqx.tree_at(f, self, value)
