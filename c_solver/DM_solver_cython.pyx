import numpy as np
cimport numpy as np

from cython.operator cimport dereference as deref
from libcpp.pair cimport pair

import scipy.signal

import matplotlib.pyplot as plt
import cython

from cyarma_lib.cyarma cimport Mat, np2vec, np2cx_mat, np2cx_vec, np2cx_cube, cx_mat, cx_cube, mat2np, cx_mat2np, cx_cube2np
from DM_solver_cython cimport DM_solver_calc_engine, noise_specifier

NO_NOISE = 0
LINDBLAD_NOISE = 1
STATIC_NOISE = 2
SPECTRUM_NOISE = 3

class noise_specs:
	def __init__(self):
		self.noise_type = 0
		
		self.STD_omega = np.array()
		self.STD_static_noise = 0

		self.Lindblad_A_i = 0
		self.Lindblad_Gamma = 0


cdef class DM_solver_core:
	cdef DM_solver_calc_engine* DM_obj
	cdef np.ndarray times
	cdef int steps
	cdef double endtime

	def __cinit__(self, int size, int steps, double endtime):
		'''
		Args:
			size : size of the Hilbert space.
			steps : number of steps in the calculation.
			endtime : end time of the simulation (in s)
		'''
		self.DM_obj = new DM_solver_calc_engine(size)
		self.steps = steps
		self.endtime = endtime

	def __dealloc__(self):
		del self.DM_obj

	def add_H1(self, np.ndarray[ np.complex_t, ndim=2 ] input_matrix, np.ndarray[ np.complex_t, ndim=1 ] input_list, int signal_type, noise_spec = None):
		cdef noise_specifier noise_specifier_obj
		cdef np.ndarray[ double, ndim=1 ] STD_omega_np

		noise_specifier_obj.noise_type = NO_NOISE
		if noise_spec is not None:
#			print("noise type: ",noise_spec.noise_type)
			if noise_spec.noise_type is STATIC_NOISE or noise_spec.noise_type is SPECTRUM_NOISE or noise_spec.noise_type is SPECTRUM_NOISE + STATIC_NOISE:
				noise_specifier_obj.noise_type += STATIC_NOISE
				noise_specifier_obj.STD_static = noise_spec.get_STD_static(self.steps, self.steps/self.endtime)
#				print("static noise: ",noise_specifier_obj.STD_static)

			if noise_spec.noise_type is SPECTRUM_NOISE or noise_spec.noise_type is SPECTRUM_NOISE + STATIC_NOISE:
				noise_specifier_obj.noise_type += SPECTRUM_NOISE
				STD_omega_np =  noise_spec.get_fft_components(self.steps, self.steps/self.endtime)
#				print("(l,l-1,h-1,h) freq amplitude element Pyx:", [STD_omega_np[0],STD_omega_np[1],STD_omega_np[-2],STD_omega_np[-1]])
#				print("number of elements in amplitude Pyx:", STD_omega_np.size)
				noise_specifier_obj.STD_omega = np2vec(STD_omega_np)
		
		self.DM_obj.add_H1(np2cx_mat(input_matrix),np2cx_vec(input_list), signal_type, noise_specifier_obj)

	def add_lindbladian(self, np.ndarray[ np.complex_t, ndim=2 ] A, double gamma):
		self.DM_obj.add_lindbladian(np2cx_mat(A), gamma)

	def calculate_evolution(self, np.ndarray[np.complex_t, ndim=2] psi0, double endtime, int steps, int iterations = 1):
		self.DM_obj.set_number_of_evalutions(iterations)
		self.DM_obj.calculate_evolution(np2cx_mat(psi0), endtime, steps)
		self.times = np.linspace(0, endtime, steps+1)

	def return_expectation_values(self, np.ndarray[np.complex_t, ndim=3] operators):
		cdef Mat[double] expect = self.DM_obj.return_expectation_values(np2cx_cube(operators))
		cdef np.ndarray[np.float64_t, ndim =2] output = None
		output = mat2np(expect, output)
		return output

	def plot_expectation(self, operators, label,number=0):
		expect = self.return_expectation_values(operators)

		plt.figure(number)
		for i in range(len(expect)):
			plt.plot(self.times*1e9, expect[i], label=label[i])
			plt.xlabel('Time (ns)')
			plt.ylabel('Population (%)/Expectation')
			plt.legend()


	def get_unitary(self):
		cdef cx_cube unitary = self.DM_obj.get_unitaries()
		cdef np.ndarray[np.complex_t, ndim =3] output = None
		output = cx_cube2np(unitary, output)
		return output

	def get_last_density_matrix(self):
		cdef cx_mat density = self.DM_obj.get_last_density_matrix()
		cdef np.ndarray[np.complex_t, ndim =2] output = None
		output = cx_mat2np(density, output)
		return output

	def get_all_density_matrices(self):
		cdef cx_cube density =self.DM_obj.get_all_density_matrices()
		cdef np.ndarray[np.complex_t, ndim =3] output = None
		output = cx_cube2np(density, output)
		return output

