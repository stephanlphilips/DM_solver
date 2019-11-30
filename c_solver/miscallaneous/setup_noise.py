#!python
#cython: language_level=3

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy


packages = find_packages()
print('packages: %s' % packages)

extensions = [
	Extension("c_solver.noise_testing",
		include_dirs=[numpy.get_include(),"./c_solver","./c_solver/solver_cpp",'.'],
		sources=["c_solver/noise_testing.pyx", 'c_solver/solver_cpp/noise_functions.cpp'], 
		language="c++",
		libraries=["armadillo","gomp",],
	  ),
	Extension("cyarma_lib.cyarma",
		include_dirs=[numpy.get_include(),"./cyarma_lib"],
		sources=["cyarma_lib/cyarma.pyx",], 
		language="c++",
		libraries=["armadillo"],
	  )
]



setup(name="c_solver",
        version="1.1",
        packages = find_packages(),
        ext_modules = cythonize(extensions)
        )
