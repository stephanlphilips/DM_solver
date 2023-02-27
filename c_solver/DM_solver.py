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
		self.noise_channel_counter_static = 0
		self.noise_correlation_matrix_static = None
		self.noise_channel_counter_dynamic = 0
		self.noise_correlation_matrix_dynamic = None
		self.activate_noise_correlation = False
		self.low_freq_cutoff_noise = None
	def add_H0(self, matrix, amplitude):
		'''
		add a constant hamiltonian to the system

		Args:
			matrix (np.ndarray[dtype=complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli Z matrix)
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
			matrix (np.ndarray[dtype=complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
			H_pulse (pulse) : pulse sequence that is related to the given matrix element.
		'''
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.NORMAL)
		self.hamiltonian_data += H_data

	def add_H1_exp(self, matrix, H_pulse):
		'''
		add a time dependent Hamiltonian to the system, where the values in H_pulse will be exponentiated before the matrix evolution will be executed
		(e.g. to simulate a voltage pulse on the tunnel coupling).

		Args:
			matrix (np.ndarray[dtype=complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
			H_pulse (pulse) : pulse sequence that is related to the given matrix element.
		'''
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.EXP)
		self.hamiltonian_data += H_data

	def add_H1_expsat(self, matrix, H_pulse):
		'''
		add a time dependent Hamiltonian to the system, where the values in H_pulse will be exponentiated with saturation before the matrix evolution will be executed
		(e.g. to simulate a voltage pulse on the tunnel coupling).

		Args:
			matrix (np.ndarray[dtype=complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
			H_pulse (pulse) : pulse sequence that is related to the given matrix element.
		'''
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.EXPSAT)
		self.hamiltonian_data += H_data

	def add_H1_tanh(self, matrix, H_pulse):
		'''
		add a time dependent Hamiltonian to the system, where the values in H_pulse will be exponentiated with saturation before the matrix evolution will be executed
		(e.g. to simulate a voltage pulse on the tunnel coupling).

		Args:
			matrix (np.ndarray[dtype=complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
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
			matrix (np.ndarray[dtype=complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
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
			matrix (np.ndarray[dtype=complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
			H_pulse (pulse) : pulse sequence that is related to the given matrix element.
		'''
		H_data = hamiltonian_data(matrix, H_pulse,signal_type.SWO2)
		self.hamiltonian_data += H_data

	def add_H1_RWA(self, matrix, H_pulse):
		'''
		add a time dependent Hamiltonian to the system, but taking the RWA approximation. Make sure to set pulse type in MW pulse to is_RWA=true.
		Args:
			matrix (np.ndarray[dtype=complex, ndim=2]) : matrix element of the Hamiltonian (e.g. Pauli X matrix)
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
			matrix (np.ndarray[dtype=complex, ndim=2]) : input matrix on what the noise needs to act.
			spectral_power_density (lamda) : function describing S(omega) (frequency expected in 2pi*f)
			A_noise_power (double) : the noise power to provide.
			TODO (later) H_pulse (pulse) : pulse describing a modulation of the noise. Optional variable
		'''
		spectrum = lambda u, x=spectral_power_density: x(u)*A_noise_power
		my_noise = noise_desciption(SPECTRUM_NOISE, spectrum, 0)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.NORMAL, noise = my_noise)
		self.hamiltonian_data += H_data
		self.noise_channel_counter_dynamic +=1

	def add_noise_static(self, matrix, T2, H_pulse=None):
		'''
		add static noise model

		Args:
			matrix (np.ndarray[dtype=complex, ndim=2]) : input matrix on what the noise needs to act.
			T2 (double) : the T2 you which you want to provide.
			TODO (later) H_pulse (pulse) : pulse describing a modulation of the noise. Optional variable
		'''
		my_noise = noise_desciption(STATIC_NOISE, None, 2./(2.*np.pi*T2)**2)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.NORMAL, noise = my_noise)
		self.hamiltonian_data += H_data
		self.noise_channel_counter_static +=1


	def add_noise_generic_exp(self, matrix, spectral_power_density, A_noise_power, H_pulse=None):
		'''
		add generic noise model

		same as add_noise_generic, but for a exponentially varying hamiltonian, see docs.
		'''
		spectrum = lambda u, x=spectral_power_density: x(u)*A_noise_power
		my_noise = noise_desciption(SPECTRUM_NOISE, spectrum, 0)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.EXP, noise = my_noise)
		self.hamiltonian_data += H_data
		self.noise_channel_counter_dynamic +=1

	def add_noise_static_exp(self, matrix, gamma):
		'''
		add static noise model

		same as add_noise_static, but for a exponentially varying hamiltonian, see docs.
		'''
# 		my_noise = noise_desciption(STATIC_NOISE, None, 2/(2*np.pi*T2)**2)
		my_noise = noise_desciption(STATIC_NOISE, None, gamma)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.EXP, noise = my_noise)
		self.hamiltonian_data += H_data
		self.noise_channel_counter_static +=1

	def add_noise_generic_expsat(self, matrix, spectral_power_density, A_noise_power, H_pulse=None,):
		'''
		add generic noise model

		same as add_noise_generic, but for a exponentially varying hamiltonian with saturation, see docs.
		'''
		spectrum = lambda u, x=spectral_power_density: x(u)*A_noise_power
		my_noise = noise_desciption(SPECTRUM_NOISE, spectrum, 0)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.EXPSAT, noise = my_noise)
		self.hamiltonian_data += H_data
		self.noise_channel_counter_dynamic +=1

	def add_noise_static_expsat(self, matrix, gamma, noise_channel = None):
		'''
		add static noise model

		same as add_noise_static, but for a exponentially varying hamiltonian with saturation, see docs.
		'''
# 		my_noise = noise_desciption(STATIC_NOISE, None, 2./(2.*np.pi*T2)**2)
		my_noise = noise_desciption(STATIC_NOISE, None, gamma)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.EXPSAT, noise = my_noise)
		self.hamiltonian_data += H_data
		self.noise_channel_counter_static +=1
	
	
	def add_noise_generic_tanh(self, matrix, spectral_power_density, A_noise_power, H_pulse=None):
		'''
		add generic noise model

		same as add_noise_generic, but for a tanh varying hamiltonian with saturation, see docs.
		'''
		spectrum = lambda u, x=spectral_power_density: x(u)*A_noise_power
		my_noise = noise_desciption(SPECTRUM_NOISE, spectrum, 0)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.TANH, noise = my_noise)
		self.hamiltonian_data += H_data
		self.noise_channel_counter_dynamic +=1
	
	
	def add_noise_static_tanh(self, matrix, gamma):
		'''
		add static noise model

		same as add_noise_static, but for a tanh varying hamiltonian with saturation, see docs.
		'''
# 		my_noise = noise_desciption(STATIC_NOISE, None, 2./(2.*np.pi*T2)**2)
		my_noise = noise_desciption(STATIC_NOISE, None, gamma)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.TANH, noise = my_noise)
		self.hamiltonian_data += H_data
		self.noise_channel_counter_static +=1
	
	
	def add_noise_generic_heis1(self, matrix, spectral_power_density, A_noise_power, H_pulse=None):
		'''
		add generic noise model

		same as add_noise_generic, but for a exponentially varying hamiltonian with saturation, see docs.
		'''
		spectrum = lambda u, x=spectral_power_density: x(u)*A_noise_power
		my_noise = noise_desciption(SPECTRUM_NOISE, spectrum, 0)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.SWO1, noise = my_noise)
		self.hamiltonian_data += H_data
		self.noise_channel_counter_dynamic +=1

	def add_noise_static_heis1(self, matrix, gamma):
		'''
		add static noise model

		same as add_noise_static, but for a exponentially varying hamiltonian with saturation, see docs.
		'''
# 		my_noise = noise_desciption(STATIC_NOISE, None, 2./(2.*np.pi*T2)**2)
		my_noise = noise_desciption(STATIC_NOISE, None, gamma)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.SWO1, noise = my_noise)
		self.hamiltonian_data += H_data
		self.noise_channel_counter_static +=1

	def add_noise_generic_heis2(self, matrix, spectral_power_density, A_noise_power, H_pulse=None):
		'''
		add generic noise model

		same as add_noise_generic, but for a exponentially varying hamiltonian with saturation, see docs.
		'''
		spectrum = lambda u, x=spectral_power_density: x(u)*A_noise_power
		my_noise = noise_desciption(SPECTRUM_NOISE, spectrum, 0)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.SWO2, noise = my_noise)
		self.hamiltonian_data += H_data
		self.noise_channel_counter_dynamic +=1

	def add_noise_static_heis2(self, matrix, gamma):
		'''
		add static noise model

		same as add_noise_static, but for a exponentially varying hamiltonian with saturation, see docs.
		'''
# 		my_noise = noise_desciption(STATIC_NOISE, None, 2./(2.*np.pi*T2)**2)
		my_noise = noise_desciption(STATIC_NOISE, None, gamma)
		H_data = hamiltonian_data(matrix, pulse(), signal_type.SWO2, noise = my_noise)
		self.hamiltonian_data += H_data
		self.noise_channel_counter_static +=1
	
	def add_noise_correlation_matrix(self, matrix_static, matrix_dynamic):
		'''
		adds correlation to the noise channel (experimental mode)
		Careful use. Best only used directly before calculate_evolution.
		No check if matrices have correct dimension.
		'''
		if not np.all(np.abs(matrix_static)<=1):
			print("Static correlation cannot be larger than 1 or smaller than -1.")
			return None    
		if not np.all(np.abs(matrix_dynamic)<=1):
			print("Dynamic correlation cannot be larger than 1 or smaller than -1.")
			return None 
		if not np.all(np.imag(matrix_dynamic)==0):
			print("Real matric expected.")
			return None
		self.activate_noise_correlation = True
		self.noise_correlation_matrix_static = matrix_static
		self.noise_correlation_matrix_dynamic = matrix_dynamic

	def set_noise_low_frequency_cutoff(self, cut_off_frequency):
		'''
		Changes the standard low-frequency cutoff frequency used for integrating out the spectrum noise. 
		This noise is then added as static noise to the Hamiltonian.
		'''
		self.low_freq_cutoff_noise = cut_off_frequency

	def calculate_evolution(self, psi0, endtime=None, steps=100000, iterations = 1):
		if endtime is None:
			# auto calculat the needed time, TODO
			raise NotImplementedError
		self.DM_solver_core = DM_solver_core(self.hamiltonian_data.size, steps, endtime*1e-9)
		
		for h_data in self.hamiltonian_data:
			if self.low_freq_cutoff_noise is not None:
				h_data.noise.low_freq_cutoff = self.low_freq_cutoff_noise
			self.DM_solver_core.add_H1(np.array(h_data.matrix, dtype=complex),
					np.array(h_data.pulse_data.get_pulse_raw(endtime, steps/endtime*1e9), dtype=complex),
					h_data.signal_type, h_data.noise)

		for i in self.lindlad_noise_terms:
			self.DM_solver_core.add_lindbladian(i[0], i[1])

		if self.activate_noise_correlation is True:
			self.DM_solver_core.add_noise_correlation(self.noise_correlation_matrix_static,self.noise_correlation_matrix_dynamic)
		self.DM_solver_core.calculate_evolution(psi0, endtime*1e-9, steps, iterations)
		self.times = np.linspace(0, endtime*1e-9, steps + 1)

	def plot_pop(self, size = 4,label = ["dd", "du", "ud", "uu"]):
		states = []
		for it in range(size):
			states.append(np.array(list(qt.basis(4,it)*qt.basis(4,it).dag()))[:,0])
		operators = np.array(states,dtype=complex) #
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
	H0 = np.zeros([2,2], dtype=complex)
	H0[0,0] = 0.5
	H0[1,1] = -0.5
	
	H1 = np.zeros([2,2], dtype=complex)
	H1[0,0] = 0.5
	
	H2 = np.zeros([2,2], dtype=complex)
	H2[1,1] = 0.5
	
	H3 = np.identity(2, dtype=complex)
	
	test.add_H0(H0*2.*np.pi, 1e9)
	oneoverfnoise=lambda omega: 1/2/np.pi/omega
	test.add_noise_generic(H1*2.*np.pi,oneoverfnoise, 3.6*1e12)
	test.add_noise_generic(H2*2.*np.pi,oneoverfnoise, 3.6*1e12)
	# corresponds to 100ns T2*
	test.set_noise_low_frequency_cutoff(1e5)
# 	test.add_noise_static(H3, 2e-8)

# 	oneoverfnoise=lambda omega: 1/2/np.pi/omega
# 	whitenoise=lambda omega: 0.*omega + 1
# 	# test.add_noise_generic(H1, whitenoise, 1e2)
# 	# test.add_noise_generic(H1, oneoverfnoise, 0.2e11)
# 	# p = pulse()
# 	# p.add_block(0,100,10.1)
# 	# test.add_H1_exp(np.eye(4), p)
# 	DM = np.zeros([44,44], dtype=complex)
# 	DM[0,0] = 1
# 	# DM[0,1] = 0.5
# 	# DM[1,0] = 0.5
# 	# DM[1,1] = 0.5

# 	#jumpdown_opertor = np.zeros([4,4], dtype=complex)
# 	#jumpdown_opertor[1,0] = 1
# 	#noise_ampl = 1/100e-9

# 	#test.add_noise_Lindblad(jumpdown_opertor, noise_ampl)
	DM = np.ones([2,2], dtype=complex)/2.
	total_time = 250
	corr = np.array([[1,0],[-1,0]],dtype=np.double)
#	corr = np.identity(2)
	test.add_noise_correlation_matrix(corr,corr)
	test.calculate_evolution(DM,total_time,total_time * 10,1000)
# 	print(test.get_unitary())
# 	uni = test.get_unitary()
	expect , time = test.return_expectation_values_general([DM])
# %%
	def exp_decay(t, freq, t2, alpha):
		return (1+np.cos(2.*np.pi*freq*t)*np.exp(-0.5*(t/t2)**alpha))/2.

	from scipy.optimize import curve_fit

	popt, pcov = curve_fit(exp_decay, time*1e-9, expect[0], p0=[1e9, 1e-7, 2])
	popt_show = [popt[0]*1e-9,popt[1]*1e9,popt[2]]
	print("Best fit parameters: ", popt_show)
	plt.figure()
	plt.plot(time, expect[0])
	plt.plot(time, exp_decay(time*1e-9, 0, popt[1], popt[2]), label='fit: f=%5.3f, T2=%5.3f, a=%5.3f' % tuple(popt_show))
	plt.legend()
	plt.show()
