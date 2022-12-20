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
    packages=find_packages(exclude=["tests", ".github", "docs"], include=["jaxdf*"]),
)
