# Density matrix Solver #

## Introduction ##
This is a simple program that solves the time evolution of a density matrix, given a time dependent Hamiltonian. I tried to add the most convenient functions that you will find in experiments (pulse shapes, pink/white noise), but feel free to extend this. To speed up the execution of the code, it it written in c++ (decent multi-threading and defined data types). Not to worry though! The code is cythonized, which means that you can fully access the library from python.
## Requirements and installation ##
Requirements:
* [Armadillo](http://arma.sourceforge.net/download.html) (version >= 1.7)
* [Openblas](http://www.openblas.net/) <br/> Note that you can also use INTEL MKL if you like.,
* [Open MPI](https://www.open-mpi.org/)
* [GNU Scientific Library](https://www.gnu.org/software/gsl/)
* [GOMP](https://gcc.gnu.org/projects/gomp/)	

Installation:
* Compile and install all the c++ libraries as given in the requirements. In Linux you can install most of them probably with your package manager. On Windows I can imagine that the installation might be mode cumbersome.,
* Compile the python code by:
```bash
python -m py_compile setup.py
```

You can than import the module by typing:
```python
import sys
sys.path.append('folder/whereever/you/saved/the/setup/file')
import DM_solver
```
## User Guide ##
The main idea of the module is that you have to specify in python your own class that generates the signals. An example where this is done is specified in `example.py`. In the following I quickly explain the main functionality you can use to generate and preform operations with/on your Hamiltonians.

The solver is initiated by typing:
```python
my_solver_object = DM_solver.VonNeumann(N)
```
Where N is the size of the Hilbert space. This creates the c object that does the solving. In the following, I'll go through all the functions to construct Hamiltonians.

### Time independent part
```python
my_numpy_hamiltonian = np.qeye(5, dtype=np.complex)
my_solver_object.add_H0(my_numpy_hamiltonian)
```
Note that the matrix must be a 128 bit complex matrix, as done in the example. Numpy will only do this by default if you matrix contains something complex. 
If you add multiple elements (`add_H0()`) they will just be summed.

### Time dependent part
#### Add a list of points (not the recommended method).
This will represent a function A(t)*H_{input}. Where the number of points must be exactly the same as the number of points used in simulation. The input matrix is the matrix that is multiplied with the points (e.g. a sigma_x*I matrix). The code needed to input such a matrix can be given by:
```python
amplitude_list = np.linspace(0,1e9,10000,dtype=np.complex)
my_solver_object.add_H1_list(input_matrix, amplitude_list)
```
#### AWG pulses
An AWG pulse can be added by using the command below. Parameters are the amplitude, rise time and start and stop time of the pulse. To generate a rise effect I used the Fermi-Dirac functions. The input matrix has the same format as mentioned before.
```python
my_solver_object.add_H1_AWG(input_matrix, amp, rise_time, start, stop)
```
#### Microwaves
To add microwaves there are basically two approaches. For a block pulse you can use:
```python
my_solver_object.add_H1_MW_RF(input_matrix, rabi_f, phase, frequency, start, stop)
```
Where the input matrix is your rotation matrix. `rabi_f` is the rabi frequency, note that for some reason this has an offset of a factor 10 (if you see why, please tell me). `phase` is the phase of the incoming signal. `frequency` is the frequency. Note that the input is not a cosine, but e**(i*\omega t). This was done since it is the most convenient for making rotating frames. You can still make a cos out of the exponential functions.

The other way of adding microwaves is by using the microwave object. In this object you can for example specify pulse shaped. At the moment only Gaussians are supported. In the following an example piece of code is given. Here a pulse is send, where you put over it a Gaussian envelope. The envelope has as center the middle of the pulse. out of the pulse times the amplitude of the signal is 0.
```python
mw_obj_1 = DM_solver.microwave_RWA()
mw_obj_1.init(rabi_f/(2*np.pi), phase, freq_RF-f_qubit, t_start, t_stop)
mw_obj_1.add_gauss_mod(sigma_Gauss) # sigma is here the standard deviation of the Gaussian distribution
mw_obj_2 = DM_solver.microwave_RWA()
mw_obj_2.init(rabi_f/(2*np.pi), phase, freq_RF-f_qubit2, t_start, t_stop)
mw_obj_2.add_gauss_mod(sigma_Gauss)
my_solver_object.add_H1_MW_RF_obj(H_mw_qubit_1, mw_obj_1)
my_solver_object.add_H1_MW_RF_obj(H_mw_qubit_2, mw_obj_2)
```
#### Global time dependency
When you have some parts of you Hamiltonian that continuously oscillate, you can add a global time dependency. This will be added latest to you Hamiltonian when constructing it. This means it will also be added on top of the noise you add. This can be a handy feature when you have the transform your Hamiltonian.

In the following a example is given of how to add a time decency to a parameter. The decency is given by:
```
paramter(t)*e^(i*2pi*f)
```
Where f is the frequency of the oscillations. Example:
```python
# This adds a time depend parameter to location (1,4) in the matrix.
# Note that the matrix is by nature hermitian, so you do not have to specify (4,1)
# Make sure the data types are set as here.
locations_1 = np.array([[1,4]],dtype=np.int32)
locations_2 = np.array([[2,4]], dtype=np.int32)
my_solver_object.add_cexp_time_dep(locations_1, (f_qubit1-f_qubit2)/2)
my_solver_object.add_cexp_time_dep(locations_2, (f_qubit2-f_qubit1)/2)
```

### Noise functions
In the noise department we have two flavors, static noise and 1/f noise.

#### Static noise

#### 1/f noise

### Running the solver
The solver will start whenever you call the following:
```python
my_solver_object.calculate_evolution(my_init_densitymatrix, t_start, t_stop, numberofsteps)
```
### Getting your results
To get the unitary representing your operation, you can type:
```python
my_solver_object.get_unitary()
time_points_sim = my_solver_oject.times
```
To get expectation values for a certain property you can type:
```python
my_solver_object.return_expectation_values(operators)
```
where operators is a matrix of dimension 3, meaning a list of you operators (see example provided).


To plot the expectation values, you can use:
```python
my_solver_object.plot_expectation(operators, label, figure_number)
plt.show()
```
For more info see the example
### Clear memory
If you run many loops, you might see that python's garbage collector does not automatically delete the c object created that solves the Von Neumann equation. The memory can be freed up by calling:
```python
my_solver_object.clear()
```
