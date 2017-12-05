from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# to_compile = "Bessel.cpp Butterworth.cpp ChebyshevI.cpp Custom.cpp Biquad.cpp Cascade.cpp  Elliptic.cpp  Legendre.cpp  Param.cpp RBJ.cpp  State.cpp  ChebyshevII.cpp  Design.cpp  Filter.cpp PoleFilter.cpp  RootFinder.cpp"
# to_compile = to_compile.split(' ')
# print(to_compile)
DSP = ["lib/DSP/source/Bessel.cpp","lib/DSP/source/Butterworth.cpp","lib/DSP/source/ChebyshevI.cpp","lib/DSP/source/Custom.cpp",
"lib/DSP/source/Biquad.cpp","lib/DSP/source/Cascade.cpp","lib/DSP/source/Elliptic.cpp","lib/DSP/source/Legendre.cpp",
"lib/DSP/source/Param.cpp","lib/DSP/source/RBJ.cpp","lib/DSP/source/State.cpp","lib/DSP/source/ChebyshevII.cpp",
"lib/DSP/source/Design.cpp","lib/DSP/source/Filter.cpp","lib/DSP/source/PoleFilter.cpp","lib/DSP/source/RootFinder.cpp"]

setup(
    name = 'ME_solver',
    ext_modules=[ 
    Extension("ME_solver", 
        include_dirs=["./lib","./lib/DSP/include"],
        sources=["lib/python_wrapper.pyx"] +DSP , 
        language="c++",
        libraries=["openblas", "armadillo", 'gomp'],
        extra_compile_args=['-fopenmp'],
      ),
    ],
    cmdclass = {'build_ext': build_ext},
)