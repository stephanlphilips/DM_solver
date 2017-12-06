Installation 
============

Requirements
-------------
Before you can compile the library, you will need to install the following dependencies:

* `Armadillo`_
.. _Armadillo: http://arma.sourceforge.net/download.html
* `Openblas`_ (Note that you can also use INTEL MKL if you like)
.. _Openblas: http://www.openblas.net/
* `Open MPI`_
.. _Open MPI: https://www.open-mpi.org/
* `GOMP`_
.. _GOMP: https://gcc.gnu.org/projects/gomp/
* `DSP lib`_ (Already included, no need to install)
.. _DSP lib: https://github.com/vinniefalco/DSPFilters/
Installation
-------------
Compile and install all the c++ libraries as given in the requirements. In Linux you can install most of them probably with your package manager. On Windows I can imagine that the installation might be mode cumbersome.

Once you installed the dependencies, you can compile the program using the python interpreter: ::
	
	python -m py_compile setup.py

You can than import the module by typing: ::

	import sys
	sys.path.append('folder/whereever/you/saved/the/setup/file')
	import DM_solver