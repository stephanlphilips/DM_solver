from c_solver.pulse_generation.pulse_generic import pulse
from c_solver.DM_solver_core import DM_solver_core, NO_NOISE, STATIC_NOISE, SPECTRUM_NOISE
from c_solver.utility.data_objects import noise_desciption, hamilotian_manager, hamiltonian_data
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

class signal_type():
	NORMAL = 1
	RWA = 2
	EXP = 3
	EXPSAT = 4
	SWO1 = 5
	SWO2 = 6
	TANH = 7

class DM_solver(object):
	"""docstring for DM_solver"""
	def __init__(self):
		self.hamiltonian_data = hamilotian_manager()
		self.lindlad_noise_terms = list()
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
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.NORMAL)
		self.hamiltonian_data += H_data

	def add_H1(self, matrix, H_pulse):
		'''
		add a time dependent hamiltonian to the system

		Args:
			matrix (np.ndarray[dtype=np.complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
			H_pulse (pulse) : pulse sequence that is related to the given matrix element.
		'''
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.NORMAL)
		self.hamiltonian_data += H_data

	def add_H1_exp(self, matrix, H_pulse):
		'''
		add a time dependent Hamiltonian to the system, where the values in H_pulse will be exponentiated before the matrix evolution will be executed
		(e.g. to simulate a voltage pulse on the tunnel coupling).

		Args:
			matrix (np.ndarray[dtype=np.complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
			H_pulse (pulse) : pulse sequence that is related to the given matrix element.
		'''
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.EXP)
		self.hamiltonian_data += H_data

	def add_H1_expsat(self, matrix, H_pulse):
		'''
		add a time dependent Hamiltonian to the system, where the values in H_pulse will be exponentiated with saturation before the matrix evolution will be executed
		(e.g. to simulate a voltage pulse on the tunnel coupling).

		Args:
			matrix (np.ndarray[dtype=np.complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
			H_pulse (pulse) : pulse sequence that is related to the given matrix element.
		'''
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.EXPSAT)
		self.hamiltonian_data += H_data

	def add_H1_tanh(self, matrix, H_pulse):
		'''
		add a time dependent Hamiltonian to the system, where the values in H_pulse will be exponentiated with saturation before the matrix evolution will be executed
		(e.g. to simulate a voltage pulse on the tunnel coupling).

		Args:
			matrix (np.ndarray[dtype=np.complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
			H_pulse (pulse) : pulse sequence that is related to the given matrix element.
		'''
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.TANH)
		self.hamiltonian_data += H_data


	def add_H1_heis1(self, matrix, H_pulse):
		'''
		add a time dependent Hamiltonian to the system, where the values in H_pulse 
		will be applied to the second order exchange interaction function 1/(1-e^2) 
		before the matrix evolution will be executed
		(e.g. to simulate a voltage pulse on the detuning coupling).

		Args:
			matrix (np.ndarray[dtype=np.complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
			H_pulse (pulse) : pulse sequence that is related to the given matrix element.
		'''
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.SWO1)
		self.hamiltonian_data += H_data

	def add_H1_heis2(self, matrix, H_pulse):
		'''
		add a time dependent Hamiltonian to the system, where the values in H_pulse 
		will be applied to the fourth order exchange interaction function (1+e^2)/(1-e^2)^3 
		before the matrix evolution will be executed. Note that this term only adds the 
		fourth order term thus should always used in combination with add_H1_heis1
		(e.g. to simulate a voltage pulse on the detuning coupling).

		Args:
			matrix (np.ndarray[dtype=np.complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
			H_pulse (pulse) : pulse sequence that is related to the given matrix element.
		'''
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.SWO2)
		self.hamiltonian_data += H_data

	def add_H1_RWA(self, matrix, H_pulse):
		'''
		add a time dependent Hamiltonian to the system, but taking the RWA approximation. Make sure to set pulse type in MW pulse to is_RWA=true.
		Args:
			matrix (np.ndarray[dtype=np.complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
			H_pulse (pulse) : pulse sequence that is related to the given matrix element.
		'''
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.RWA)
		self.hamiltonian_data += H_data

	def add_noise_Lindblad(self, operator, rate):
		'''
		add lindblad term to the system:

		Args:
			operator (np.ndarray) : jump operator for the noise
			rate (double) : rate at which to apply the operator (e.g. 1/T1)
		'''
		self.lindlad_noise_terms.append((operator, np.sqrt(rate)))

	def add_noise_generic(self, matrix, spectral_power_density, A_noise_power, H_pulse=None):
		'''
		add generic noise model

		Args:
			matrix (np.ndarray[dtype=np.complex, ndim=2]) : input matrix on what the noise needs to act.
			spectral_power_density (lamda) : function describing S(omega) (frequency expected in 2pi*f)
			A_noise_power (double) : the noise power to provide.
			TODO (later) H_pulse (pulse) : pulse describing a modulation of the noise. Optional variable
		'''
		spectrum = lambda u, x=spectral_power_density: x(u)*A_noise_power
		my_noise = noise_desciption(SPECTRUM_NOISE, spectrum, 0)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.NORMAL, noise = my_noise)
		self.hamiltonian_data += H_data

	def add_noise_static(self, matrix, T2, H_pulse=None):
		'''
		add static noise model

		Args:
			matrix (np.ndarray[dtype=np.complex, ndim=2]) : input matrix on what the noise needs to act.
			T2 (double) : the T2 you which you want to provide.
			TODO (later) H_pulse (pulse) : pulse describing a modulation of the noise. Optional variable
		'''
		my_noise = noise_desciption(STATIC_NOISE, None, 2./(2.*np.pi*T2)**2)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.NORMAL, noise = my_noise)
		self.hamiltonian_data += H_data


	def add_noise_generic_exp(self, matrix, spectral_power_density, A_noise_power, H_pulse=None):
		'''
		add generic noise model

		same as add_noise_generic, but for a exponentially varying hamiltonian, see docs.
		'''
		spectrum = lambda u, x=spectral_power_density: x(u)*A_noise_power
		my_noise = noise_desciption(SPECTRUM_NOISE, spectrum, 0)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.EXP, noise = my_noise)
		self.hamiltonian_data += H_data

	def add_noise_static_exp(self, matrix, gamma):
		'''
		add static noise model

		same as add_noise_static, but for a exponentially varying hamiltonian, see docs.
		'''
# 		my_noise = noise_desciption(STATIC_NOISE, None, 2/(2*np.pi*T2)**2)
		my_noise = noise_desciption(STATIC_NOISE, None, gamma)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.EXP, noise = my_noise)
		self.hamiltonian_data += H_data

	def add_noise_generic_expsat(self, matrix, spectral_power_density, A_noise_power, H_pulse=None):
		'''
		add generic noise model

		same as add_noise_generic, but for a exponentially varying hamiltonian with saturation, see docs.
		'''
		spectrum = lambda u, x=spectral_power_density: x(u)*A_noise_power
		my_noise = noise_desciption(SPECTRUM_NOISE, spectrum, 0)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.EXPSAT, noise = my_noise)
		self.hamiltonian_data += H_data

	def add_noise_static_expsat(self, matrix, gamma):
		'''
		add static noise model

		same as add_noise_static, but for a tanh varying hamiltonian with saturation, see docs.
		'''
# 		my_noise = noise_desciption(STATIC_NOISE, None, 2./(2.*np.pi*T2)**2)
		my_noise = noise_desciption(STATIC_NOISE, None, gamma)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.EXPSAT, noise = my_noise)
		self.hamiltonian_data += H_data
	
	
	def add_noise_generic_tanh(self, matrix, spectral_power_density, A_noise_power, H_pulse=None):
		'''
		add generic noise model

		same as add_noise_generic, but for a tanh varying hamiltonian with saturation, see docs.
		'''
		spectrum = lambda u, x=spectral_power_density: x(u)*A_noise_power
		my_noise = noise_desciption(SPECTRUM_NOISE, spectrum, 0)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.TANH, noise = my_noise)
		self.hamiltonian_data += H_data
	
	
	def add_noise_static_tanh(self, matrix, gamma):
		'''
		add static noise model

		same as add_noise_static, but for a tanh varying hamiltonian with saturation, see docs.
		'''
# 		my_noise = noise_desciption(STATIC_NOISE, None, 2./(2.*np.pi*T2)**2)
		my_noise = noise_desciption(STATIC_NOISE, None, gamma)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.TANH, noise = my_noise)
		self.hamiltonian_data += H_data
	
	
	def add_noise_generic_heis1(self, matrix, spectral_power_density, A_noise_power, H_pulse=None):
		'''
		add generic noise model

		same as add_noise_generic, but for a exponentially varying hamiltonian with saturation, see docs.
		'''
		spectrum = lambda u, x=spectral_power_density: x(u)*A_noise_power
		my_noise = noise_desciption(SPECTRUM_NOISE, spectrum, 0)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.SWO1, noise = my_noise)
		self.hamiltonian_data += H_data

	def add_noise_static_heis1(self, matrix, gamma):
		'''
		add static noise model

		same as add_noise_static, but for a exponentially varying hamiltonian with saturation, see docs.
		'''
# 		my_noise = noise_desciption(STATIC_NOISE, None, 2./(2.*np.pi*T2)**2)
		my_noise = noise_desciption(STATIC_NOISE, None, gamma)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.SWO1, noise = my_noise)
		self.hamiltonian_data += H_data

	def add_noise_generic_heis2(self, matrix, spectral_power_density, A_noise_power, H_pulse=None):
		'''
		add generic noise model

		same as add_noise_generic, but for a exponentially varying hamiltonian with saturation, see docs.
		'''
		spectrum = lambda u, x=spectral_power_density: x(u)*A_noise_power
		my_noise = noise_desciption(SPECTRUM_NOISE, spectrum, 0)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.SWO2, noise = my_noise)
		self.hamiltonian_data += H_data

	def add_noise_static_heis2(self, matrix, gamma):
		'''
		add static noise model

		same as add_noise_static, but for a exponentially varying hamiltonian with saturation, see docs.
		'''
# 		my_noise = noise_desciption(STATIC_NOISE, None, 2./(2.*np.pi*T2)**2)
		my_noise = noise_desciption(STATIC_NOISE, None, gamma)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.SWO2, noise = my_noise)
		self.hamiltonian_data += H_data

	def calculate_evolution(self, psi0, endtime=None, steps=100000, iterations = 1):
		if endtime is None:
			# auto calculat the needed time, TODO
			raise NotImplementedError
		self.DM_solver_core = DM_solver_core(self.hamiltonian_data.size, steps, endtime*1e-9)

		for h_data in self.hamiltonian_data:
			self.DM_solver_core.add_H1(np.array(h_data.matrix, dtype=np.complex),
					np.array(h_data.pulse_data.get_pulse_raw(endtime, steps/endtime*1e9), dtype=np.complex),
					h_data.signal_type, h_data.noise)

		for i in self.lindlad_noise_terms:
			self.DM_solver_core.add_lindbladian(i[0], i[1])
		self.DM_solver_core.calculate_evolution(psi0, endtime*1e-9, steps, iterations)
		self.times = np.linspace(0, endtime*1e-9, steps + 1)

	def plot_pop(self):
		dd = np.array(list(qt.basis(4,0)*qt.basis(4,0).dag()))[:,0]
		du = np.array(list(qt.basis(4,1)*qt.basis(4,1).dag()))[:,0]
		ud = np.array(list(qt.basis(4,2)*qt.basis(4,2).dag()))[:,0]
		uu = np.array(list(qt.basis(4,3)*qt.basis(4,3).dag()))[:,0]
		operators = np.array([dd,du,ud,uu],dtype=complex) #
		label = ["dd", "du", "ud", "uu"]
		expect = self.DM_solver_core.return_expectation_values(operators)
		number =0

		plt.figure(number)
		for i in range(len(expect)):
			plt.plot(self.times*1e9, expect[i], label=label[i])
			plt.xlabel('Time (ns)')
			plt.ylabel('Population (%)/Expectation')
			plt.legend()
		plt.show()

	def plot_expect(self):
		XI = np.array(list(qt.basis(4,1)*qt.basis(4,3).dag() + qt.basis(4,0)*qt.basis(4,2).dag() + qt.basis(4,2)*qt.basis(4,0).dag() + qt.basis(4,3)*qt.basis(4,1).dag()))[:,0]
		IX = np.array(list(qt.basis(4,0)*qt.basis(4,1).dag() + qt.basis(4,1)*qt.basis(4,0).dag() + qt.basis(4,2)*qt.basis(4,3).dag() + qt.basis(4,3)*qt.basis(4,2).dag()))[:,0]
		XX = np.array(list(qt.basis(4,0)*qt.basis(4,3).dag() + qt.basis(4,1)*qt.basis(4,2).dag() + qt.basis(4,2)*qt.basis(4,1).dag() + qt.basis(4,3)*qt.basis(4,0).dag()))[:,0]
		ZZ = np.array(list(qt.basis(4,0)*qt.basis(4,0).dag() - qt.basis(4,1)*qt.basis(4,1).dag() - qt.basis(4,2)*qt.basis(4,2).dag() + qt.basis(4,3)*qt.basis(4,3).dag()))[:,0]
		ZI = np.array(list(qt.basis(4,0)*qt.basis(4,0).dag() + qt.basis(4,1)*qt.basis(4,1).dag() - qt.basis(4,2)*qt.basis(4,2).dag() - qt.basis(4,3)*qt.basis(4,3).dag()))[:,0]
		IZ = np.array(list(qt.basis(4,0)*qt.basis(4,0).dag() - qt.basis(4,1)*qt.basis(4,1).dag() + qt.basis(4,2)*qt.basis(4,2).dag() - qt.basis(4,3)*qt.basis(4,3).dag()))[:,0]
		YY = qt.tensor(qt.sigmay(), qt.sigmay())[:,:]

		operators = np.array([ZI,IZ,ZZ,XI,IX,XX],dtype=complex)
# 		operators = np.array([ZI,IZ,ZZ],dtype=complex)
		label = ["ZI", "IZ", "ZZ", "XI", "IX", "XX","YY"]
		expect = self.DM_solver_core.return_expectation_values(operators)

		plt.figure()
		for i in range(len(expect)):
			plt.plot(self.times*1e9, expect[i], label=label[i])
			plt.xlabel('Time (ns)')
			plt.ylabel('Population (%)/Expectation')
			plt.legend()
		plt.show()

	def return_expectation_values(self):
		XI = np.array(list(qt.basis(4,1)*qt.basis(4,3).dag() + qt.basis(4,0)*qt.basis(4,2).dag() + qt.basis(4,2)*qt.basis(4,0).dag() + qt.basis(4,3)*qt.basis(4,1).dag()))[:,0]
		IX = np.array(list(qt.basis(4,0)*qt.basis(4,1).dag() + qt.basis(4,1)*qt.basis(4,0).dag() + qt.basis(4,2)*qt.basis(4,3).dag() + qt.basis(4,3)*qt.basis(4,2).dag()))[:,0]
		XX = np.array(list(qt.basis(4,0)*qt.basis(4,3).dag() + qt.basis(4,1)*qt.basis(4,2).dag() + qt.basis(4,2)*qt.basis(4,1).dag() + qt.basis(4,3)*qt.basis(4,0).dag()))[:,0]
		ZZ = np.array(list(qt.basis(4,0)*qt.basis(4,0).dag() - qt.basis(4,1)*qt.basis(4,1).dag() - qt.basis(4,2)*qt.basis(4,2).dag() + qt.basis(4,3)*qt.basis(4,3).dag()))[:,0]
		ZI = np.array(list(qt.basis(4,0)*qt.basis(4,0).dag() + qt.basis(4,1)*qt.basis(4,1).dag() - qt.basis(4,2)*qt.basis(4,2).dag() - qt.basis(4,3)*qt.basis(4,3).dag()))[:,0]
		IZ = np.array(list(qt.basis(4,0)*qt.basis(4,0).dag() - qt.basis(4,1)*qt.basis(4,1).dag() + qt.basis(4,2)*qt.basis(4,2).dag() - qt.basis(4,3)*qt.basis(4,3).dag()))[:,0]
		YY = qt.tensor(qt.sigmay(), qt.sigmay())[:,:]
		XXYY = (qt.tensor(qt.sigmax(), qt.sigmax())/2+qt.tensor(qt.sigmay(), qt.sigmay())/2)[:,:]
		ZIIZ = (ZI-IZ)/2.
		proj = np.array(list(qt.basis(4,3)*qt.basis(4,2).dag() + qt.basis(4,2)*qt.basis(4,3).dag()))[:,0]

		operators = np.array([ZI,IZ,ZZ,XI,IX,XX,YY,XXYY,ZIIZ,proj],dtype=complex)

		label = ["ZI", "IZ", "ZZ", "XI", "IX", "XX","YY","XXYY","ZI-IZ","proj"]
		expect = self.DM_solver_core.return_expectation_values(operators)
		# time in ns
		return expect, self.times*1e9, label
	
	def return_expectation_values_general(self,op_list):
		operators = np.array(op_list,dtype=complex)

		expect = self.DM_solver_core.return_expectation_values(operators)
		# time in ns
		return expect, self.times*1e9

	def get_unitary(self):
		list_unitary = self.DM_solver_core.get_unitary()
		# time in ns
		return list_unitary

	def get_density_matrices(self):
		list_dm_matrices = self.DM_solver_core.get_all_density_matrices()
		# time in ns
		return list_dm_matrices

	def get_last_density_matrix(self):
		list_dm_matrix = self.DM_solver_core.get_last_density_matrix()
		# time in ns
		return list_dm_matrix

if __name__ == '__main__':
	test = DM_solver()
	H1 = np.zeros([4,4], dtype=np.complex)
	H1[0,0] = 0.5
	H1[1,1] = -0.5
	H1[2,2] = 0.5
	H1[3,3] = -0.5
	# print(H1)
	# test.add_H0(np.eye(4, dtype=np.complex), 1)
	test.add_H0(H1, 1e9*2*np.pi)
	M = np.eye(4, dtype=np.complex)
	
	# test.add_noise_static(H1, 2e-8)

	oneoverfnoise=lambda omega: 1/2/np.pi/omega
	whitenoise=lambda omega: 0.*omega + 1
	# test.add_noise_generic(H1, whitenoise, 1e2)
	# test.add_noise_generic(H1, oneoverfnoise, 0.2e11)
	# p = pulse()
	# p.add_block(0,100,10.1)
	# test.add_H1_exp(np.eye(4), p)
	DM = np.zeros([4,4], dtype=np.complex)
	DM[0,0] = 1
	# DM[0,1] = 0.5
	# DM[1,0] = 0.5
	# DM[1,1] = 0.5

	#jumpdown_opertor = np.zeros([4,4], dtype=complex)
	#jumpdown_opertor[1,0] = 1
	#noise_ampl = 1/100e-9

	#test.add_noise_Lindblad(jumpdown_opertor, noise_ampl)

	test.calculate_evolution(DM, 100,8000,3)
	uni = test.get_unitary()
	print(uni)
	# test.plot_expect()
	# expect , time, label = test.return_expectation_values()

	# def exp_decay(t, freq, gamma, alpha):
	# 	return np.cos(2*np.pi*freq*t)*np.exp(-(gamma*t)**alpha)

	# from scipy.optimize import curve_fit

	# popt, pcov = curve_fit(exp_decay, time*1e-9, expect[4], p0=[1e9, 800e5, 1])
	# print(popt)

	# plt.plot(time, expect[4])
	# plt.plot(time, exp_decay(time*1e-9, *popt), label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
	#plt.show()
