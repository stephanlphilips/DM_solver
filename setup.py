#!python
#cython: language_level=3

from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy


packages = find_packages()
print('packages: %s' % packages)

setup(name="DM_solver",
        version="1.2",
        packages = find_packages(),
        )
