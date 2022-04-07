from DM_solver.pulse_generation.pulse import pulse
from DM_solver.lib.python_c_interface import python_c_interface

import matplotlib.pyplot as plt
import numpy as np

class H_channel():
	def __init__(self, matrix):
		self.matrix    = matrix
		self.noise     = []
		self.pulse     = pulse()
		self.pulse_mod = lambda x : x

		self.__pulse_cache = None
		self.__hamiltonian = None
	
	def add_noise(self, noise_source):
		self.noise.append(noise_source)

	def add_pulse_modulator(self, func):
		self.pulse_mod = func

	def plot_pulse(self, t_end = None, sample_rate=10e9, noise=True):
		if t_end is None:
			t_end = self.pulse.pulse_data.total_time

		pulse = self.render_pulse(t_end, sample_rate, cache=False)
		
		self.__pulse_cache = None
		
		t = np.linspace(0, t_end, pulse.size)

		plt.plot(t, pulse)
		plt.xlabel('Time (s)')
		plt.ylabel('Amplitude (a.u.)')
		plt.show()

	def render_pulse(self, t_end, sample_rate, cache = True):
		if self.__pulse_cache is None:
			self.__pulse_cache = self.pulse.render(t_end, sample_rate)

		noise_data = np.zeros(self.__pulse_cache.shape)
		for noise in self.noise:
			noise_data += noise.render_noise(round(t_end*sample_rate), sample_rate)
		
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

	def calculate(self, psi_0, end_time, sample_rate, n_iter = 1):
		self.__t_end = end_time
		self.__sample_rate = sample_rate

		self.c_interface = python_c_interface(psi_0, end_time, sample_rate)
		for i in range(n_iter):
			print(f'running iteration {i}')
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
