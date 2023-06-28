# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

