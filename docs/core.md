# `jaxdf.core`

## Module Overview

This module is the fundamental part of the `jaxdf` framework.

At its core is the `Field` class, a key element of `jaxdf`. This class is designed as a module derived from [`equinox.Module`](https://github.com/patrick-kidger/equinox), which means it's a JAX-compatible dataclass. All types of discretizations within `jaxdf` are derived from the `Field` class.

Another crucial feature of `jaxdf` is the `operator` decorator. This decorator enables the implementation of multiple-dispatch functionality through the [`plum`](https://github.com/wesselb/plum) library. This is particularly useful for creating new operators within the framework.

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
