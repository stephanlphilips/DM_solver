from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name = 'ME_solver',
    ext_modules=[ 
    Extension("ME_solver", 
        sources=["lib/python_wrapper.pyx"], 
        language="c++",
        libraries=["openblas", "armadillo", "gsl", 'gomp'],
        extra_compile_args=['-fopenmp', '-lopenblas','-lgsl','-lgomp'],
      ),
    ],
    cmdclass = {'build_ext': build_ext},
)