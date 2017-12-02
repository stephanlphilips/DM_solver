import numpy as np
cimport numpy as np

from cython.operator cimport dereference as deref
from libcpp.pair cimport pair

import scipy.signal

import matplotlib.pyplot as plt

include "/usr/lib/python3.6/site-packages/cyarma/cyarma.pyx"


cdef extern from "VonNeuman_core.h":
	cdef cppclass VonNeumannSolver:
		VonNeumannSolver(double)
		void add_H0(cx_mat)
		void add_H1_list(cx_mat,cx_vec)
		void add_H1_AWG(Mat[double], cx_mat)
		void add_H1_AWG(Mat[double], cx_mat, cube)
		void add_H1_MW(cx_mat,double, double, double, double, double)
		void add_H1_MW_obj(cx_mat,phase_microwave_RWA)
		void add_H1_element_dep_f(cx_mat, int, int, cx_mat)
		void add_magnetic_noise(cx_mat, double)
		void add_noise_object(noise)
		void add_1f_noise(cx_mat, double, double)
		void mk_param_time_dep(Mat[int], double)
		void set_number_of_evalutions(int)
		void calculate_evolution(cx_mat, double, double, int)
		Mat[double] return_expectation_values(cx_cube)
		cx_mat get_unitary()
		cx_mat get_lastest_rho()
		cx_cube get_all_density_matrices()

	cdef cppclass phase_microwave_RWA:
		void init(double,double,double,double,double)
		void add_gauss_amp_mod(double)

	cdef cppclass noise:
		void init(cx_mat, double)
		void init(cx_mat, double, double)
		void add_param_dep(pair[int,int], cx_mat)
		void add_param_matrix_dep(cx_mat, pair[int,int], cx_mat)

cdef class microwave_RWA:
	cdef phase_microwave_RWA* MW_RWA_obj

	def __cinit__(self):
		self.MW_RWA_obj = new phase_microwave_RWA()
	def init(self, double amp, double phi, double fre, double t_start,double t_stop):
		self.MW_RWA_obj.init( amp, phi, fre, t_start, t_stop)
	def add_gauss_mod(self,double sigma):
		self.MW_RWA_obj.add_gauss_amp_mod(sigma)
	cdef phase_microwave_RWA return_object(self):
		return deref(self.MW_RWA_obj)

cdef class noise_py:
	cdef noise* noise_obj

	def __cinit__(self):
		self.noise_obj = new noise()

	def init(self, np.ndarray[np.complex_t, ndim=2] input_matrix, double T2):
		self.noise_obj.init(np2cx_mat(input_matrix), T2)

	def init(self, np.ndarray[np.complex_t, ndim=2] input_matrix, double noise_amplitude, double alpha):
		self.noise_obj.init(np2cx_mat(input_matrix), noise_amplitude, alpha)

	def add_param_dep(self, tuple locations, np.ndarray[np.complex_t, ndim=2] function_parmaters):
		self.noise_obj.add_param_dep(locations, np2cx_mat(function_parmaters))

	def add_param_matrix_dep(self, np.ndarray[np.complex_t, ndim=2] input_matrix, tuple locations, np.ndarray[np.complex_t, ndim=2] function_parmaters):
		self.noise_obj.add_param_matrix_dep(np2cx_mat(input_matrix), locations, np2cx_mat(function_parmaters))

	cdef noise return_object(self):
		return deref(self.noise_obj)

cdef class VonNeumann:
	cdef VonNeumannSolver* Neum_obj
	cdef np.ndarray times
	def __cinit__(self, double size):
		self.Neum_obj = new VonNeumannSolver(size)

	def clear(self):
		# Needed since python considers this not as garbage for some reason when deleted from an enclosing class ...
		del self.Neum_obj

	def add_H0(self, np.ndarray[ np.complex_t, ndim=2 ] input_matrix):
		self.Neum_obj.add_H0(np2cx_mat(input_matrix))

	def add_H1_list(self, np.ndarray[ np.complex_t, ndim=2 ] input_matrix, np.ndarray[ np.complex_t, ndim=1 ] input_list):
		self.Neum_obj.add_H1_list(np2cx_mat(input_matrix),np2cx_vec(input_list))

	def add_H1_AWG(self, np.ndarray[ np.double_t, ndim=2 ] time_input, np.ndarray[ np.complex_t, ndim=2 ] input_matrix, np.ndarray[ np.double_t, ndim=3 ] filters = None):
		'''
		Adds AWG pulse,
		Time_input: array with timings (see manual)
		input matrix: matrix elements that must be pulsed.
		filtering: filtering elements, format: (e.g. buttherworth/Bessel filters.)
		[ [type, order_filter , fc] , [ ... ]]
		E.g. for Tek, 
		[ ['Butt', 1, 300e6], 
		  ['Bessel', 2, 380e6]
		]
		'''

		# Pulse without filtering
		# if filtering is None:
		self.Neum_obj.add_H1_AWG(np2arma(time_input), np2cx_mat(input_matrix))
		# else:
		# 	my_filter = []

		# 	for i in filtering:


		# self.Neum_obj.add_H1_AWG(np2arma(time_input), np2cx_mat(input_matrix), np2cube(my_filtering))

	def add_H1_MW_RF(self, np.ndarray[ np.complex_t, ndim=2 ] input_matrix, double rabi_f, double phase, double frequency, double start, double stop):
		self.Neum_obj.add_H1_MW(np2cx_mat(input_matrix), rabi_f, phase, frequency, start, stop)

	def add_H1_MW_RF_obj(self, np.ndarray[ np.complex_t, ndim=2 ] input_matrix, microwave_RWA my_mwobj):
		self.Neum_obj.add_H1_MW_obj(np2cx_mat(input_matrix), my_mwobj.return_object())

	def add_H1_element_dep_f(self, np.ndarray[np.complex_t, ndim=2] input_matrix, int i , int j, np.ndarray[np.complex_t, ndim=2] matrix_param):
		self.Neum_obj.add_H1_element_dep_f(np2cx_mat(input_matrix), i ,j, np2cx_mat(matrix_param))

	def add_magnetic_noise(self, np.ndarray[ np.complex_t, ndim=2 ] input_matrix, double T2):
		self.Neum_obj.add_magnetic_noise(np2cx_mat(input_matrix), T2)

	def add_magnetic_noise_obj(self, noise_py noise_obj):
		self.Neum_obj.add_noise_object(noise_obj.return_object());

	def add_1f_noise(self, np.ndarray[ np.complex_t, ndim=2 ] input_matrix, double noise_strength, double alpha=1.):
		self.Neum_obj.add_1f_noise(np2cx_mat(input_matrix), noise_strength, alpha)
	
	def add_cexp_time_dep(self, np.ndarray[int, ndim=2] locations, double frequency):
		self.Neum_obj.mk_param_time_dep(np2arma(locations), frequency)

	def set_number_of_evalutions(self, int number):
		self.Neum_obj.set_number_of_evalutions(number)

	def calculate_evolution(self, np.ndarray[np.complex_t, ndim=2] psi0, double start, double stop, int steps):
		self.times = np.linspace(start,stop, steps+1)
		self.Neum_obj.calculate_evolution(np2cx_mat(psi0), start, stop, steps)

	def get_times(self):
		return self.times

	def return_expectation_values(self, np.ndarray[np.complex_t, ndim=3] operators):
		cdef Mat[double] expect = self.Neum_obj.return_expectation_values(np2cx_cube(operators))
		cdef np.ndarray[np.float64_t, ndim =2] output = None
		output = mat2np(expect, output)
		return output

	def plot_expectation(self, operators, label,number=0):
		expect = self.get_expectation(operators)

		plt.figure(number)
		for i in range(len(expect)):
			plt.plot(self.times*1e9, expect[i], label=label[i])
			plt.xlabel('Time (ns)')
			plt.ylabel('Population (%)/Expectation')
			plt.legend()

	def get_expectation(self,operators):
		cdef Mat[double] output = self.Neum_obj.return_expectation_values(np2cx_cube(operators))
		cdef np.ndarray[np.float64_t, ndim =2] expect = None
		expect = mat2np(output, expect)

		return expect

	def get_unitary(self):
		cdef cx_mat unitary = self.Neum_obj.get_unitary()
		cdef np.ndarray[np.complex_t, ndim =2] output = None
		output = cx_mat2np(unitary, output)
		return output

	def get_lastest_density_matrix(self):
		cdef cx_mat density = self.Neum_obj.get_lastest_rho()
		cdef np.ndarray[np.complex_t, ndim =2] output = None
		output = cx_mat2np(density, output)
		return output

	def get_all_density_matrices(self):
		cdef cx_cube density =self.Neum_obj.get_all_density_matrices()
		cdef np.ndarray[np.complex_t, ndim =3] output = None
		output = cx_cube2np(density, output)
		return output
