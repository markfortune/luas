# File: setup.py
from setuptools import find_packages, setup

setup(
    name="luas", version = "0.0.1",
    description='Python implementation of 2D Gaussian processes using JAX',
    author='Mark Fortune',
    author_email='fortunma@tcd.ie',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
