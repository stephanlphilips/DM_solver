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
		self.noise_spectral_density_power = 0
		self.S_omega_sqrt = np.array()
		self.sigma_static_noise = 0
		self.Lindblad_A_i = 0
		self.Lindblad_Gamma = 0


cdef class DM_solver_core:
	cdef DM_solver_calc_engine* DM_obj
	cdef np.ndarray times

	def __cinit__(self, double size):
		self.DM_obj = new DM_solver_calc_engine(size)

	def __dealloc__(self):
		del self.DM_obj

	def add_H1(self, np.ndarray[ np.complex_t, ndim=2 ] input_matrix, np.ndarray[ np.complex_t, ndim=1 ] input_list, int signal_type, noise_spec = None):
		cdef noise_specifier noise_specifier_obj
		cdef np.ndarray[ double, ndim=1 ] spectral_density

		noise_specifier_obj.noise_type = NO_NOISE
		if noise_spec is not None:
			if noise_spec.type is STATIC_NOISE or noise_spec.type is SPECTRUM_NOISE + STATIC_NOISE:
				noise_specifier_obj.noise_type += STATIC_NOISE
				noise_specifier_obj.T2 = noise_spec.T2

			if noise_spec.type is SPECTRUM_NOISE or noise_spec.type is SPECTRUM_NOISE + STATIC_NOISE:
				noise_specifier_obj.noise_type += SPECTRUM_NOISE
				S_omega_sqrt =  np.array(noise_spec.S_omega_sqrt, dtype =np.double)
				noise_specifier_obj.S_omega_sqrt = np2vec(S_omega_sqrt)
				noise_specifier_obj.noise_power = noise_spec.noise_power
		
		self.DM_obj.add_H1(np2cx_mat(input_matrix),np2cx_vec(input_list), signal_type, noise_specifier_obj)

	def calculate_evolution(self, np.ndarray[np.complex_t, ndim=2] psi0, double endtime, int steps):
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
		cdef cx_mat unitary = self.DM_obj.get_unitary()
		cdef np.ndarray[np.complex_t, ndim =2] output = None
		output = cx_mat2np(unitary, output)
		return output

	def get_lastest_density_matrix(self):
		cdef cx_mat density = self.DM_obj.get_lastest_rho()
		cdef np.ndarray[np.complex_t, ndim =2] output = None
		output = cx_mat2np(density, output)
		return output

	def get_all_density_matrices(self):
		cdef cx_cube density =self.DM_obj.get_all_density_matrices()
		cdef np.ndarray[np.complex_t, ndim =3] output = None
		output = cx_cube2np(density, output)
		return output

