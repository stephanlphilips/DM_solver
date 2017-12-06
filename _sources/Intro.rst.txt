Introduction
==============

This is a simple program that solves the time evolution of a density matrix, given a time dependent Hamiltonian. I tried to add the most convenient functions that you will find in experiments (pulse shapes, pink/white noise), but feel free to extend this. To speed up the execution of the code, it it written in c++ (decent multi-threading and defined data types). Not to worry though! The code is cythonized, which means that you can fully access the library from python.

This modules solves the Von Neumann equation:

.. math::

	\rho (t + \Delta t ) = U^{\dagger} \rho (t) U


Where :math:`\Delta t` will be determined by the number of points calculated in the simulation. :math:`U` is given by

.. math::
	
	U = e^{iH \Delta t}

Where :math:`H` is the hamiltonian at time t. This module contains a toolbox that allows to easily make H time dependent.