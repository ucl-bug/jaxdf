# jaxdf - JAX-based Discretization Framework

[![Support](https://dcbadge.vercel.app/api/server/VtUb4fFznt?style=flat)](https://discord.gg/VtUb4fFznt)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![codecov](https://codecov.io/gh/ucl-bug/jaxdf/branch/main/graph/badge.svg?token=FIUYOCFDYL)](https://codecov.io/gh/ucl-bug/jaxdf)
[![CI](https://github.com/ucl-bug/jaxdf/actions/workflows/tests.yml/badge.svg)](https://github.com/ucl-bug/jaxdf/actions/workflows/tests.yml)

[**Overview**](#overview)
| [**Example**](#example)
| [**Installation**](#installation)
| [**Documentation**](https://ucl-bug.github.io/jaxdf/)
| [**Support**](#support)

<br/>

## Overview

Jaxdf is a package based on [JAX](https://jax.readthedocs.io/en/stable/) that provides a coding framework for creating differentiable numerical simulators with arbitrary discretizations.

The primary objective of Jaxdf is to aid in the construction of numerical models for physical systems, like wave propagation, or the numerical resolution of partial differential equations, in a manner that is easily tailored to the user's research requirements. These models are pure functions that can be seamlessly integrated into arbitrary differentiable programs written in [JAX](https://jax.readthedocs.io/en/stable/). For instance, they can be employed as layers within neural networks, or utilized in constructing a physics loss function.


<br/>

## Example

The script below constructs the non-linear operator **(∇<sup>2</sup> + sin)**, applying a Fourier spectral discretization on a square 2D domain. It then utilizes this operator to define a loss function. The gradient of this loss function is calculated using JAX's Automatic Differentiation.


```python
from jaxdf import operators as jops
from jaxdf import FourierSeries, operator
from jaxdf.geometry import Domain
from jax import numpy as jnp
from jax import jit, grad


# Defining operator
@operator
def custom_op(u, *, params=None):
  grad_u = jops.gradient(u)
  diag_jacobian = jops.diag_jacobian(grad_u)
  laplacian = jops.sum_over_dims(diag_jacobian)
  sin_u = jops.compose(u)(jnp.sin)
  return laplacian + sin_u

# Defining discretizations
domain = Domain((128, 128), (1., 1.))
parameters = jnp.ones((128,128,1))
u = FourierSeries(parameters, domain)

# Define a differentiable loss function
@jit
def loss(u):
  v = custom_op(u)
  return jnp.mean(jnp.abs(v.on_grid)**2)

gradient = grad(loss)(u) # gradient is a FourierSeries
```

<br/>

## Installation

Before proceeding with the installation of `jaxdf`, ensure that [JAX is already installed](https://github.com/google/jax#installation) on your system. If you intend to utilize `jaxdf` with NVidia GPU support, follow the instructions to install JAX accordingly.

To install `jaxdf` from PyPI, use the `pip` command:

```bash
pip install jaxdf
```

For development purposes, install `jaxdf` by either cloning the repository or downloading and extracting the compressed archive. Afterward, navigate to the root folder in a terminal, and execute the following command:
```bash
pip install --upgrade poetry
poetry install
```
This will install the dependencies and the package itself (in editable mode).


## Support

[![Support](https://dcbadge.vercel.app/api/server/VtUb4fFznt?style=flat)](https://discord.gg/VtUb4fFznt)

If you encounter any issues with the code or wish to suggest new features, please feel free to open an issue. If you seek guidance, wish to discuss something, or simply want to say hi, don't hesitate to write a message in our [Discord channel](https://discord.gg/VtUb4fFznt).


<br/>

## Contributing

Contributions are absolutely welcome! Most contributions start with an issue. Please don't hesitate to create issues in which you ask for features, give feedback on performances, or simply want to reach out.

To make a pull request, please look at the detailed [Contributing guide](CONTRIBUTING.md) for how to do it, but fundamentally keep in mind the following main guidelines:

- If you add a new feature or fix a bug:
  - Make sure it is covered by tests
  - Add a line in the changelog using `kacl-cli`
- If you changed something in the documentation, make sure that the documentation site can be correctly build using `mkdocs serve`

<br/>

<br/>

## Citation

[![arXiv](https://img.shields.io/badge/arXiv-2111.05218-b31b1b.svg?style=flat)](https://arxiv.org/abs/2111.05218)

An initial version of this package was presented at the [Differentiable Programming workshop](https://diffprogramming.mit.edu/) at NeurIPS 2021.

```bibtex
@article{stanziola2021jaxdf,
    author={Stanziola, Antonio and Arridge, Simon and Cox, Ben T. and Treeby, Bradley E.},
    title={A research framework for writing differentiable PDE discretizations in JAX},
    year={2021},
    journal={Differentiable Programming workshop at Neural Information Processing Systems 2021}
}
```

<br/>


#### Acknowledgements

- Some of the packaging of this repository is done by editing [this templace from @rochacbruno](https://github.com/rochacbruno/python-project-template)
- The multiple-dispatch method employed is based on `plum`, check out this amazing project: https://github.com/wesselb/plum

#### Related projects

1. [`odl`](https://github.com/odlgroup/odl) Operator Discretization Library (ODL) is a python library for fast prototyping focusing on (but not restricted to) inverse problems.
3. [`deepXDE`](https://deepxde.readthedocs.io/en/latest/): a TensorFlow and PyTorch library for scientific machine learning.
4. [`SciML`](https://sciml.ai/): SciML is a NumFOCUS sponsored open source software organization created to unify the packages for scientific machine learning.
