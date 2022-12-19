"""Python setup.py for jaxdf package"""
import io
import os

from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("jaxdf", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


def read_variable_from_py(variable_name, path):
    """Read a variable from a Python file.
    >>> read_variable_from_py("VERSION", "jaxdf/__about__.py")
    '0.1.0'
    """
    return read(path).split(variable_name + '="')[1].split('"')[0]


setup(
    name="jaxdf",
    version=read_variable_from_py("VERSION", "jaxdf/__about__.py"),
    description="A JAX-based research framework for writing differentiable numerical simulators with arbitrary discretizations",
    url="https://github.com/ucl-bug/jaxdf",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Antonio Stanziola, UCL BUG",
    author_email="a.stanziola@ucl.ac.uk",
    packages=find_packages(exclude=["tests", ".github", "docs"]),
    install_requires=read_requirements(".requirements/requirements.txt"),
    extras_require={
        "test": read_requirements(".requirements/requirements-test.txt"),
        "dev": read_requirements(".requirements/requirements-dev.txt"),
        "doc": read_requirements(".requirements/requirements-doc.txt"),
    },
    python_requires=">=3.7",
    license="GNU Lesser General Public License (LGPL)",
    keywords=[
        "jax",
        "pde",
        "discretization",
        "differential-equations",
        "simulation",
        "differentiable-programming",
    ],
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
