from setuptools import setup, find_packages

# Get version
_dct = {}
with open("jaxdf/version.py") as f:
    exec(f.read(), _dct)
__version__ = _dct["__version__"]

setup(
    name="jaxdf",
    version=__version__,
    description="A JAX-based research framework for writing differentiable numerical simulators with arbitrary discretizations",
    author="Antonio Stanziola, UCL BUG",
    author_email="a.stanziola@ucl.ac.uk",
    packages=find_packages(exclude=["docs"]),
    package_data={"jaxdf": ["py.typed"]},
    python_requires=">=3.7",
    install_requires=open("_setup/requirements.txt", "r").readlines(),
    extras_require={
        "dev": open("_setup/dev_requirements.txt", "r").readlines(),
    },
    url="https://bug.medphys.ucl.ac.uk/",
    license="GNU Lesser General Public License (LGPL)",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ],
)
