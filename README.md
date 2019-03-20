DM Solver 
=========

Welcome to the gihub page of the DM (Density Matrix) solver. This is a simple program that solves the time evolution of a density matrix, given a time dependent Hamiltonian. Most of the commonly used pulseshapes in experiments are supported. Also functionality to simulate  pink/white noise is included. To speed up the execution of the code, it it written in c++ (as it has decent multi-threading and typed variables). Not to worry though! The code is cythonized, which means that you can fully access the library from python.

Requirements
-------------
Before you can compile the library, you will need to install the following dependencies:

* [Armadillo](http://arma.sourceforge.net/download.html)
* [Openblas](http://www.openblas.net/) (Note that you can also use INTEL MKL if you like)
* [Open MP](https://www.open-mpi.org/)
* [GOMP](https://gcc.gnu.org/projects/gomp/)
* [DSP lib](https://github.com/vinniefalco/DSPFilters/) (Already included, no need to install seperately)
* cython (install with pip)
* matplotlib (install with pip)



Installation
-------------
Compile and install all the c++ libraries as given in the requirements. In Linux you can install most of them probably with your package manager. On Windows I can imagine that the installation might be mode cumbersome.

Once you installed the dependencies, you can compile the program by typing the following in a command line:
```bash	
python setup.py install
```
You can than import the module by typing the following in your python script:
```python
import DM_solver
```

Note for os X:
make sure to use the gcc compiler when compiling, if you have problems, you can try running in the command line:
```bash
export CC=gcc-8
export CXX=g++-8
```

Documentation
-------------

Documentation for the library can be found at:

https://DM_solver.readthedocs.io
