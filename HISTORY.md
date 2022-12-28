# Changelog


## (latest)

### Bug Fix

* The default_parameters function now works with custom field types. [Antonio Stanziola]

* __rpow__ for OnGrid. [antonio]

* Avoids changing parameters of OnGrid inside jax transformations. [Antonio Stanziola]

* Spectral gradient for signal of even length now treats the Nyquist frequency correctly. [Antonio Stanziola]

* Jaxlib in requirements, or actions fail. [Antonio Stanziola]

* Staggering in fd kernel. [Antonio Stanziola]

* Added jaxlib to requirements for CI. [Antonio Stanziola]

* Error on 3d staggered derivatives. [Antonio Stanziola]

### Features

* Heterogeneous laplacian. [Antonio Stanziola]

* Fourier field value on arbitrary point using __call__ method. [Antonio Stanziola]

* Automatically infer missing dimension for scalar fields. [Antonio Stanziola]

* Shift operator. [Antonio Stanziola]

* Staggering for Fourier differential operators. [Antonio Stanziola]

### Refactoring

* Updated docs. [Antonio Stanziola]

* Renamed ode variable update, removed wrong test in utils. [Antonio Stanziola]

* Updated support/packaging files. [Antonio Stanziola]

### Tests

* Increase core.py coverage. [Antonio Stanziola]

* Added FourierSeries gradient tests. [Antonio Stanziola]


