# `jaxdf` Documentation

`jaxdf` is a customizable framework for writing differentiable simulators, that decouples the mathematical definition of the problem from the underlying discretization.

The underlying computations are performed using JAX, and are thus compatible with the broad set of program transformations that the package allows, such as automatic differentiation and batching. This enables rapid prototyping of multiple customized representations for a given problem, to develop physics-based neural network layers and to write custom physics losses, while maintaining the speed and flexibility required for research applications.

It also contains a growing open-source library of differentiable discretizations compatible with the JAX ecosystem.

</br>

---

⚙️ This documentation is work in progress as `jaxdf` is in active development. Expect breaking changes until a release version is published.
