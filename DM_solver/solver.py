from DM_solver.pulse_generation.pulse import pulse
from DM_solver.lib.python_c_interface import python_c_interface

import matplotlib.pyplot as plt
import numpy as np

class H_channel():
	def __init__(self, matrix, amplitude):
		self.matrix    = matrix
		self.__amplitude = amplitude
		self.noise     = []
		self.pulse     = pulse()
		self.pulse.add_block(0, -1, 1)
		self.pulse_mod = lambda x : x

		self.__pulse_cache = None
		self.__hamiltonian = None

	def add_pulse(self, pulse):
		self.pulse = pulse
	
	def add_noise(self, noise_source):
		self.noise.append(noise_source)

	def add_pulse_modulator(self, func):
		self.pulse_mod = func

	def plot_pulse(self, t_end = None, sample_rate=10e9, noise=True):
		if t_end is None:
			t_end = self.pulse.total_time

		pulse = self.render_pulse(t_end, sample_rate)
		t = np.linspace(0, t_end, pulse.size)

		plt.plot(t, pulse)
		plt.xlabel('Time (s)')
		plt.ylabel('Amplitude (a.u.)')
		plt.show()

	def render_pulse(self, t_end, sample_rate):
		if self.__pulse_cache is None:
			self.__pulse_cache = self.pulse.render(t_end, sample_rate)*self.__amplitude

		noise_data = np.zeros(self.__pulse_cache.shape)
		for noise in self.noise:
			noise_data += noise.render_noise(npt, sample_rate)
		
		return self.pulse_mod(self.__pulse_cache + noise_data)

class H_solver():
	def __init__(self):
		self.channels = []
		self.lindbladians = []
		self.c_interface = None

		self.__t_end = 0
		self.__sample_rate = 1
	
	def add_channels(self, *channels):
		self.channels += channels

	def add_lindbladian(self, matrix, gamma):
		self.lindbladians.append([matrix, gamma])

	def calculate(self, psi_0, end_time, time_step, n_iter = 1):
		self.__t_end = end_time
		self.__sample_rate = 1/time_step

		self.c_interface = python_c_interface(psi_0, end_time, 1/time_step)
		for i in range(n_iter):
			for channel in self.channels:
				self.c_interface.add_H_channel_data(channel)

			for lindbladian in self.lindbladians:
				self.c_interface.add_lindbladian(*lindbladian)

			self.c_interface.calculate_evolution()

	def return_expectation_values(self,*operator_list):
		expectation_values = []
		for op in operator_list:
			expectation_values.append(self.c_interface.get_expectation_values(op))
		
		return expectation_values

	def return_time(self):
		return np.linspace(0, self.__t_end, round(self.__t_end*self.__sample_rate)+1)[:-1]

	def get_unitary(self):
		return self.c_interface.get_unitary()


if __name__ == '__main__':
	from DM_solver.utility.pauli import X, Y, Z
	import numpy as np
	f_res  = 1e9
	f_rabi = 10e6

	# Set qubit resonance frequnecy of 1GHZ -- Sz hamiltonian.
	Qubit1_Z = H_channel(Z/2, 2*np.pi*f_res)
	# Qubit1_Z.add_noise(amp, STATIC)

	# add MW driving pulse @10MHZ -- Sx hamiltonian.
	Qubit1_X = H_channel(X/2, 2*np.pi*f_rabi)
	p = pulse()
	p.add_MW_pulse(20e-9,70e-9, amp=1, freq=f_res)
	Qubit1_X.add_pulse(p)
	# Qubit1_X.plot_pulse(50e-9, 1/10e-12)
	# Qubit1_Z.plot_pulse(50e-9, 1/10e-12)

	calculation = H_solver()
	calculation.add_channels(Qubit1_Z,Qubit1_X)

	psi_0 = np.matrix([[1,0],[0,0]], dtype=np.complex128)/2
	calculation.calculate(psi_0, end_time = 100e-9, time_step = 10e-12, n_iter=3)

	Z_expect, X_expect = calculation.return_expectation_values(Z, X)
	t = calculation.return_time()
	plt.plot(t, Z_expect)
	plt.plot(t, X_expect)
	plt.show()
		