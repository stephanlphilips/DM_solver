#!python
#cython: language_level=3

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy


packages = find_packages()
print('packages: %s' % packages)

extensions = [
	Extension("cyarma_lib.cyarma",
		include_dirs=[numpy.get_include(),"./cyarma_lib"],
		sources=["cyarma_lib/cyarma.pyx",], 
		language="c++",
		libraries=["armadillo"],
	  ),
	Extension("c_solver.DM_solver_core",
		include_dirs=[numpy.get_include(),"./c_solver","./c_solver/solver_cpp",'.' ],
		sources=["c_solver/DM_solver_cython.pyx",
					"c_solver/solver_cpp/DM_solver_core.cpp",
					'c_solver/solver_cpp/hamiltonian_constructor.cpp',
					'c_solver/solver_cpp/math_functions.cpp',
					'c_solver/solver_cpp/memory_mgmnt.cpp',
					'c_solver/solver_cpp/noise_functions.cpp'], 
		language="c++",
		libraries=["armadillo","gomp",],
		extra_compile_args=['-fopenmp'],
	  )
]



setup(name="c_solver",
        version="1.1",
        packages = find_packages(),
        ext_modules = cythonize(extensions)
        )
