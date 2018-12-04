from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

DSP = ["c_solver/DSP/source/Bessel.cpp","c_solver/DSP/source/Butterworth.cpp","c_solver/DSP/source/ChebyshevI.cpp","c_solver/DSP/source/Custom.cpp",
"c_solver/DSP/source/Biquad.cpp","c_solver/DSP/source/Cascade.cpp","c_solver/DSP/source/Elliptic.cpp","c_solver/DSP/source/Legendre.cpp",
"c_solver/DSP/source/Param.cpp","c_solver/DSP/source/RBJ.cpp","c_solver/DSP/source/State.cpp","c_solver/DSP/source/ChebyshevII.cpp",
"c_solver/DSP/source/Design.cpp","c_solver/DSP/source/Filter.cpp","c_solver/DSP/source/PoleFilter.cpp","c_solver/DSP/source/RootFinder.cpp"]


packages = find_packages()
print('packages: %s' % packages)

extensions = [
	Extension("c_solver.ME_solver",
		include_dirs=[numpy.get_include(),"./c_solver","./c_solver/DSP/include",'.'],
		sources=["c_solver/ME_solver.pyx"] + DSP, 
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
