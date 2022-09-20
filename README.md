DM Solver 
=========

Welcome to the github page of the DM (Density Matrix) solver. This is a simple program that solves the time evolution of a density matrix, given a time dependent Hamiltonian. Most of the commonly used pulseshapes in experiments are supported. Also functionality to simulate  pink/white noise is included. To speed up the execution of the code, it it written in c++. Not to worry though! The code just needs to be compiled and can be fully accessed from the python API.

Requirements
-------------
Before you can compile the library, you will need to install the following dependencies:

* [Armadillo](http://arma.sourceforge.net/download.html)
* [Openblas](http://www.openblas.net/) (Note that you can also use INTEL MKL if you like)
* [Open MP](https://www.open-mpi.org/)
* [GOMP](https://gcc.gnu.org/projects/gomp/)


Installation
-------------
Compile and install all the c++ libraries as given in the requirements. In Linux/OS X (use brew) you can install them using your package manager. On Windows the installation will be mode cumbersome.

The libraries can be compiled using the makefile by running the following command in the `DM_solver/lib/makefile` folder:
```bash	
make
```
It might by that you need to manually link the armadillo library (e.g. an error one `#include <armadillo>`), this can by done by adjusting the `CPPFLAGS` and the `LDLIBS` makefile, to the folder on your system where the library is installed (example formatting in the makefile).

If using brew in OS X, it is likely that clang will will be selected as your default compiler, this can resolved by adjusting the `CC` and `CCX` variables in the makefile, e.g. when gcc-12 is installed:
```
CC  = gcc-12
CXX = g++-12
```
Documentation
-------------

The examples in the example folder should be self explanatory.