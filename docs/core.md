
# `jaxdf.core`

The `core` module contains the basic abstractions of the `jaxdf` framework.

The key component is the `Field` class, which is a PyTree (see [`equinox`](https://github.com/patrick-kidger/equinox) and [`treeo`](https://github.com/cgarciae/treeo) for two great libraries that deal with defining JAX-compatible PyTrees from classes) from which all discretizations are defined, and the `operator` decorator which allows the use of multiple-dispatch (vial [`plum`](https://github.com/wesselb/plum)) for defining novel operators.

::: jaxdf.core
    handler: python
    selection:
        filters:
            - "!^_"
            - "__init__$"
    rendering:
        show_root_heading: true
        show_source: false
        show_object_full_path: True
