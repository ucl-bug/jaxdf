Changelog
=========


(0.2.0)
------------

New Features
~~~~~~~~~~~~
- fourier field value on arbitrary point using __call__ method.
- Shift_operator for FourierSeries. [Antonio Stanziola]
- automatically infer missing dimension for scalar fields. [Antonio
  Stanziola]
- staggering for Fourier differential operators. [Antonio
  Stanziola]

Fix
~~~
- __rpow__ for OnGrid. [antonio]
- Avoids changing parameters of OnGrid inside jax transformations.
  [Antonio Stanziola]
- Spectral gradient for signal of even length now treats the Nyquist
  frequency correctly. [Antonio Stanziola]
- Fourier laplacian
- Jaxlib in requirements, or actions fail. [Antonio Stanziola]
- Staggering in fd kernel. [Antonio Stanziola]
- Added jaxlib to requirements for CI. [Antonio Stanziola]
- Error on 3d staggered derivatives. [Antonio Stanziola]

Breaking
~~~~~~~~
- removed the `ode.py` module. [Antonio Stanziola]
- params are only accepted as keyword argument. [Antonio Stanziola]
- changed how operator parameters are returned.
  [Antonio Stanziola]

Other
~~~~~
- Remove breaking changes warning. [Antonio Stanziola]
- Black pre-commit. [Antonio Stanziola]
- Plum documentation via plumkdocs. [Antonio Stanziola]
- Params reimplemented as private property. [Antonio Stanziola]
