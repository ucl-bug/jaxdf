# jaxdf - JAX-based Discretization Framework

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![codecov](https://codecov.io/gh/astanziola/jaxdf/branch/main/graph/badge.svg?token=6J03OMVJS1)](https://codecov.io/gh/astanziola/jaxdf)
![Continous Integration](https://github.com/astanziola/jaxdf/actions/workflows/ci-build.yml/badge.svg)

[**Overview**](#overview)
| [**Installation**](#installation)

## Overview

jaxdf is a [JAX](https://jax.readthedocs.io/en/stable/)-based package defining a coding framework for writing differentiable numerical simulators with arbitrary discretizations. 

The intended use is to build numerical models of physical systems, such as wave propagation, or the numerical solution of partial differential equations, that are easy to customize to the user's research needs. Such models are pure functions that can be included into arbitray differentiable programs written in [JAX](https://jax.readthedocs.io/en/stable/). For example, they can be used as layers of neural networks, or to build a physics loss function.

## Example


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

1. [`odl`](https://github.com/odlgroup/odl) Operator Discretization Library (ODL) is a python library for fast prototyping focusing on (but not restricted to) inverse problems.
3. [`deepXDE`](https://deepxde.readthedocs.io/en/latest/): a TensorFlow and PyTorch library for scientific machine learning.
4. [`SciML`](https://sciml.ai/): SciML is a NumFOCUS sponsored open source software organization created to unify the packages for scientific machine learning. 