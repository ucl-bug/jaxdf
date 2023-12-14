# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.7] - 2023-11-24
### Changed
- The Quickstart tutorial has been upgdated.
- The property `Field.ndim` has now been moved into `Field.domain.ndim`, as it is fundamentally a property of the domain
- The `init_params` function now will inherit the default parameters from its operator, to remove any source of ambiguity. This means that it should not have any default values, and an error is raised if it does.

### Removed
- The `__about__` file has been removed, as it is redundant
- The function `params_map` is removed, use `jax.tree_util.tree_map` instead.
- Operators are now expected to return only their outputs, and not parameters. If you need to get the parameters of an operator use its `default_params` method. To minimize problems for packages relying on `jaxdf`, in this release the outputs of an `operator` are filtered to keep only the first one. This will change soon to allow the user to return arbitrary PyTrees.

### Added
- JaxDF `Field`s are now based on [equinox](https://github.com/patrick-kidger/equinox). In theory, this should allow to use `jaxdf` with all the [scientific libraries for the jax ecosystem](https://github.com/patrick-kidger/equinox#see-also-other-libraries-in-the-jax-ecosystem). In practice, please raise an issue when you encounter one of the inevitable bugs :)
- The new `operator.abstract` decorator can be used to define an unimplemented operator, for specifying input arguments and docstrings.
- `Linear` fields are now defined as equal if they have the same set of parameters and the same `Domain`.
- `Ongrid` fields now have the method `.add_dim()`, which adds an extra tailing dimension to its parameters. This **is not** an in-place update: the method returns a new field.
- The function `jaxdf.util.get_implemented` is now exposed to the user.
- Added `laplacian` operator for `FiniteDifferences` fields.
- JaxDF now uses standard Python logging. To set the logging level, use `jaxdf.logger.set_logging_level`, for example `jaxdf.logger.set_logging_level("DEBUG")`. The default level is `INFO`.
- Fields have now a handy property `.θ` which is an alias for `.params`
- `Continuous` and `Linear` fields now have the `.is_complex` property
- `Field` and `Domain` are now `Modules`s, which are based on from `equinox.Module`. They are entirely equivalent to `equinox.Module`, but have the extra `.replace` method that is used to update a single field.

### Deprecated
- The property `.is_field_complex` is now deprecated in favor of `.is_complex`. Same goes for `.is_real`.
- `Field.get_field` is now deprecated in favor of the `__call__` method.
- The `@discretization` decorator is deprecated, as now `Fields` are `equinox` modules. It is just not needed now, and until removed it will act as a simple pass-trough

### Fixed
- `OnGrid.from_grid` now automatically adds a dimension at the end of the array for scalar fields, if needed
- Added a custom operator for `equinox.internal._omega._Metaω` objects and Fields, which makes the library compatible with `diffrax`
- Fixed an error for `functional` operator on `OnGrid` fields that was returning a `PyTree` instead of a `Field`.

## [0.2.6] - 2023-06-28
### Changed
- removed `jaxlib` from dependencies. See https://github.com/google/jax/discussions/16380 for more information

## [0.2.5] - 2023-06-23
### Fixed
- The default_parameters function now works with custom field types
- `__rpow__` for `OnGrid`
- Avoids changing parameters of `OnGrid` inside jax transformations
- Spectral gradient for signal of even length now treats the Nyquist frequency correctly
- Staggering in `FiniteDifferences` kernel
- Incorrect behaviour for 3d staggered derivatives

### Added
- Heterogeneous laplacian operator
- `FourierSeries` values on arbitrary point using `__call__` method
- Automatically infer missing dimension for scalar fields
- Shift operator
- Staggering for `FourierSeries` differential operators

### Changed
- Updated docs
- Renamed ode variable update, removed wrong test in utils
- Updated support/packaging files

[Unreleased]: https://github.com/ucl-bug/jaxdf/compare/0.2.7...master
[0.2.7]: https://github.com/ucl-bug/jaxdf/compare/0.2.6...0.2.7
[0.2.6]: https://github.com/ucl-bug/jaxdf/compare/0.2.5...0.2.6
[0.2.5]: https://github.com/ucl-bug/jaxdf/tree/0.2.5

