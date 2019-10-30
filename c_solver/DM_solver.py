from dataclasses import dataclass
from c_solver.pulse_generation.pulse_generic import pulse
from c_solver.DM_solver_core import DM_solver_core
import numpy as np
import matplotlib.pyplot as plt

from qutip import basis
class signal_type():
	NORMAL = 1
	RWA = 2
	EXP = 3

@dataclass
class hamiltonian_data():
	matrix : np.ndarray
	pulse_data : pulse
	signal_type : int
	dsp : int
	noise : int

	def __eq__(self, other):
		if (self.matrix == other.matrix).all() and self.signal_type == other.signal_type:
			return True

		return False

class hamilotian_manager():
	def __init__(self):
		self.hamiltonian_data = list()
		self.size = None

	def __add__(self, other):
		if self.size is None:
			self.size = other.matrix.shape[0]
		else:
			if self.size != other.matrix.shape[0]:
				raise ValueError("Matrix of dimension {} provided while the previous matrix were of the dimension {}.".format(other.matrix.shape[0], self.size))
		for hamiltonian in self.hamiltonian_data:
			if hamiltonian == other:
				hamiltonian.pulse_data += other.pulse_data
				# TODO noise 
				return self

		self.hamiltonian_data.append(other)
		return self

	def __iter__(self):
		self._iteration = 0
		return self

	def __next__(self):
		self._iteration += 1
		if self._iteration <= len(self.hamiltonian_data):
			return self.hamiltonian_data[self._iteration-1]
		else:
			raise StopIteration

class DM_solver(object):
	"""docstring for DM_solver"""
	def __init__(self):
		self.hamiltonian_data = hamilotian_manager()
		self.DM_solver_core = None

	def add_H0(self, matrix, amplitude):
		'''
		add a constant hamiltonian to the system

		Args:
			matrix (np.ndarray[dtype=np.complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli Z matrix)
			amplitude (double) : amplitude of the matrix element (e.g. 1e7 (Hz))
		'''
		H_pulse = pulse()
		H_pulse.add_block(0,-1,amplitude)
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.NORMAL,0,0)

		self.hamiltonian_data += H_data

	def add_H1(self, matrix, H_pulse):
		'''
		add a time dependent hamiltonian to the system

		Args:
			matrix (np.ndarray[dtype=np.complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
			H_pulse (pulse) : pulse sequence that is related to the given matrix element.
		'''
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.NORMAL,0,0)
		self.hamiltonian_data += H_data

	def add_H1_exp(self, matrix, H_pulse):
		'''
		add a time dependent Hamiltonian to the system, where the values in H_pulse will be exponentiated before the matrix evolution will be executed
		(e.g. to simulate a voltage pulse on the tunnel coupling).

		Args:
			matrix (np.ndarray[dtype=np.complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
			H_pulse (pulse) : pulse sequence that is related to the given matrix element.
		'''
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.EXP,0,0)
		self.hamiltonian_data += H_data

	def add_H1_RWA(self, matrix, H_pulse):
		'''
		add a time dependent Hamiltonian to the system, where the values in H_pulse will be exponentiated before the matrix evolution will be executed
		(e.g. to simulate a voltage pulse on the tunnel coupling).

		Args:
			matrix (np.ndarray[dtype=np.complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
			H_pulse (pulse) : pulse sequence that is related to the given matrix element.
		'''
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.RWA,0,0)
		self.hamiltonian_data += H_data

	def add_noise_Lindblad(self, operator, rate, H_pulse=None):
		raise NotImplemented

	def add_noise_generic(self, matrix, spectral_power_density, H_pulse=None):
		raise NotImplemented

	def add_noise_static(self, matrix, T2, H_pulse):
		raise NotImplemented

	def add_noise_generic_exp(self, matrix, spectral_power_density, H_pulse=None):
		raise NotImplemented

	def add_noise_static_exp(self, matrix, T2):
		raise NotImplemented

	def calculate_evolution(self, psi0, endtime=None, steps=100000):
		self.DM_solver_core = DM_solver_core(self.hamiltonian_data.size)

		for h_data in self.hamiltonian_data:
			print(h_data.pulse_data.get_pulse_raw(endtime, steps/endtime*1e9))
			self.DM_solver_core.add_H1(np.array(h_data.matrix, dtype=np.complex),
					np.array(h_data.pulse_data.get_pulse_raw(endtime, steps/endtime*1e9), dtype=np.complex),
					h_data.signal_type)

		self.DM_solver_core.calculate_evolution(psi0, endtime*1e-9, steps)
		self.times = np.linspace(0, endtime*1e-9, steps + 1)

	def plot_pop(self):
		dd = np.array(list(basis(4,0)*basis(4,0).dag()))[:,0]
		du = np.array(list(basis(4,1)*basis(4,1).dag()))[:,0]
		ud = np.array(list(basis(4,2)*basis(4,2).dag()))[:,0]
		uu = np.array(list(basis(4,3)*basis(4,3).dag()))[:,0]
		operators = np.array([dd,du,ud,uu],dtype=complex) #
		label = ["dd", "du", "ud", "uu"]
		expect = self.DM_solver_core.return_expectation_values(operators)
		number =0
		plt.figure(number)
		for i in range(len(expect)):
			print(expect[i])
			plt.plot(self.times*1e9, expect[i], label=label[i])
			plt.xlabel('Time (ns)')
			plt.ylabel('Population (%)/Expectation')
			plt.legend()
		plt.show()

if __name__ == '__main__':
	test = DM_solver()
	H1 = np.zeros([4,4], dtype=np.complex)
	H1[0,3] = 1
	H1[1,1] = 1
	H1[2,2] = 1
	H1[3,0] = 1
	# print(H1)
	test.add_H0(np.eye(4, dtype=np.complex), 1)
	test.add_H0(H1/2, 1e9*2*np.pi)

	p = pulse()
	p.add_block(0,100,10.1)
	test.add_H1_exp(np.eye(4), p)
	DM = np.zeros([4,4], dtype=np.complex)
	DM[0,0] = 1

	test.calculate_evolution(DM, 10,10000)

	print(test.DM_solver_core.get_unitary())
	test.plot_pop()