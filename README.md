# jaxdf - JAX-based Discretization Framework

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![codecov](https://codecov.io/gh/ucl-bug/jaxdf/branch/main/graph/badge.svg?token=FIUYOCFDYL)](https://codecov.io/gh/ucl-bug/jaxdf)
![Continous Integration](https://github.com/ucl-bug/jaxdf/actions/workflows/ci-build.yml/badge.svg)
[![Documentation](https://github.com/ucl-bug/jaxdf/actions/workflows/build_docs.yml/badge.svg)](https://ucl-bug.github.io/jaxdf/)

[**Overview**](#overview)
| [**Example**](#example)
| [**Installation**](#installation)
| [**Documentation**](https://ucl-bug.github.io/jaxdf/)

## ⚠️ This library is still in development. Breaking changes may occur.

<br/>

## Overview

jaxdf is a [JAX](https://jax.readthedocs.io/en/stable/)-based package defining a coding framework for writing differentiable numerical simulators with arbitrary discretizations. 

The intended use is to build numerical models of physical systems, such as wave propagation, or the numerical solution of partial differential equations, that are easy to customize to the user's research needs. Such models are pure functions that can be included into arbitray differentiable programs written in [JAX](https://jax.readthedocs.io/en/stable/). For example, they can be used as layers of neural networks, or to build a physics loss function.

<br/>

## Example

The following script builds the non-linear operator **(∇<sup>2</sup> + sin)**, using a Fourier spectral discretization on a square 2D domain. The output is given over the whole collocation grid.


```python
from jaxdf import operators as jops
from jaxdf.core import operator
from jaxdf.geometry import Domain
from jax import numpy as jnp
from jax import jit, grad

# Defining operator
@operator()
def custom_op(u):
    grad_u = jops.gradient(u)
    diag_jacobian = jops.diag_jacobian(grad_u)
    laplacian = jops.sum_over_dims(mod_diag_jacobian)
    sin_u = jops.elementwise(jnp.sin)(u)
    return laplacian + sin_u

# Defining discretizations
domain = Domain((256, 256), (1., 1.))
fourier_discr = FourierSeries(domain)
u_fourier_params, u = fourier_discr.empty_field(name='u')

# Discretizing operators: getting pure functions and parameters
result = custom_op(u=u)
op_on_grid = result.get_field_on_grid()
global_params = result.get_global_params() # This contains the Fourier filters

# Compile and use the pure function
result_on_grid = jit(op_on_grid)(
    global_params,
    {"u": u_fourier_params}
)

# Define a differentiable loss function
def loss(u_params):
    op_output = jit(op_on_grid)(global_params, {"u": u_params})
    return jnp.mean(jnp.abs(op_output)**2)

gradient = grad(loss)(u_fourier_params)
```

<br/>

## Installation

Before installing `jaxdf`, make sure that [you have installed JAX](https://github.com/google/jax#installation). Follow the instruction to install JAX with NVidia GPU support if you want to use `jaxdf` on the GPUs. 

Install jaxdf by `cd` in the repo folder an run
```bash
pip install -r requirements.txt
pip install .
```

If you want to run the notebooks, you should also install the following packages
```bash
pip install jupyterlab, tqdm
```

<br/>

## Citation

This package will be presented at the [Differentiable Programming workshop](https://diffprogramming.mit.edu/) at NeurIPS 2021. 

A preprint will be uploaded to Arxiv soon.

```bibtex
@article{stanziola2021jaxdf,
    author={Stanziola, Antonio and Arridge, Simon and Cox, Ben T. and Treeby, Bradley E.},
    title={A research framework for writing differentiable PDE discretizations in JAX},
    year={2021},
    journal={Differentiable Programming workshop at Neural Information Processing Systems 2021}
}
```

<br/>

### Related projects

1. [`odl`](https://github.com/odlgroup/odl) Operator Discretization Library (ODL) is a python library for fast prototyping focusing on (but not restricted to) inverse problems.
3. [`deepXDE`](https://deepxde.readthedocs.io/en/latest/): a TensorFlow and PyTorch library for scientific machine learning.
4. [`SciML`](https://sciml.ai/): SciML is a NumFOCUS sponsored open source software organization created to unify the packages for scientific machine learning. 