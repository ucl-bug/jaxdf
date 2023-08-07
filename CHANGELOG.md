# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- The Quickstart tutorial has been upgdated.
- The `@new_discretization` decorator has been renamed `@discretization`
- The property `Field.ndim` has now been moved into `Field.domain.ndim`, as it is fundamentally a property of the domain
- Before, `OnGrid` fields were able to automatically add an extra dimension if needed at initialization. This however can easily clash with some of the internal operations of jax during compliation. This is now not possible, use `.from_grid` instead, which implements the same functionality.

### Removed
- The `__about__` file has been removed, as it is redundant
- The function `params_map` is removed, use `jax.tree_util.tree_map` instead.

### Added
- The new `operator.abstract` decorator can be used to define an unimplemented operator, with the goal of specifying input arguments and docstrings.
- `Linear` fields are now defined as equal if they have the same set of parameters.
- `Ongrid` fields now have the property `.add_dim`, which adds an extra tailing dimension to its parameters. The method returns a new field.
- The function `jaxdf.util.get_implemented` is now exposed to the user.
- Added `laplacian` operator for `FiniteDifferences` fields.

### Deprecated
- The property `.is_field_complex` is now deprecated in favor of `.is_complex`. Same argument for `.is_real`
- `Field.get_field` is now deprecated in favor of the `__call__` metho.

### Fixed
- `OnGrid.from_grid` now automatically adds a dimension at the end of the array for scalar fields, if needed
- Added a custom operator for `equinox.internal._omega._Metaω` objects and Fields, which makes the library compatible with `diffrax`

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

[Unreleased]: https://github.com/ucl-bug/jaxdf/compare/0.2.6...master
[0.2.6]: https://github.com/ucl-bug/jaxdf/compare/0.2.5...0.2.6
[0.2.5]: https://github.com/ucl-bug/jaxdf/tree/0.2.5
