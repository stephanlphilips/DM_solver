from c_solver.DM_solver_core import NO_NOISE, STATIC_NOISE, SPECTRUM_NOISE
from c_solver.pulse_generation.pulse_generic import pulse
from dataclasses import dataclass
from scipy.integrate import quad

import numpy as np
import types
import copy


@dataclass
class noise_desciption():
	noise_type : int = NO_NOISE
	spectrum : types.LambdaType = None
	STD_SQUARED : float = 0
	low_freq_cutoff : float = 0.1
	
	def __add__(self, other):
		noise_descr = copy.copy(self)
		if self.spectrum is None:
			if other.spectrum is not None:
				noise_descr.noise_type += SPECTRUM_NOISE
				noise_descr.spectrum = other.spectrum
		else:
			if other.spectrum is not None:
				spectrum = lambda u, x=self.spectrum, y=other.spectrum: x(u) + y(u)
				noise_descr.spectrum = spectrum

		if self.STD_SQUARED == 0:
			if other.STD_SQUARED != 0:
				noise_descr.noise_type += STATIC_NOISE
				noise_descr.STD_SQUARED = other.STD_SQUARED
		else:
			if other.spectrum is not None:
				noise_descr.STD_SQUARED = (np.sqrt(self.STD_SQUARED) + np.sqrt(other.STD_SQUARED))**2

		return noise_descr

	def get_fft_components(self, n_points, sample_rate):
		'''
		Args :
			n_points (int) : number of points of noise to generate.
			sample_rate (double) : sample rate of the experiment.
		
		Returns :
			STD_omega (np.ndarray<double>) : standard deviations of the noise at the requested frequencies (freq_postive). Normalized by sample frequency.
		'''

		n_points = 2*2**(int(np.log2(n_points))+1)
		frequencies = np.fft.fftfreq(n_points, d=1/sample_rate)

		# get postive frequencies (we will be taking the sqrt as we will add up real and imag components)
		freq_postive = abs(frequencies[frequencies<0])[::-1]
# 		print("freq_lower_bound 1: ",freq_postive[0])
		STD_omega = np.sqrt(self.spectrum(freq_postive*2.*np.pi))
		return STD_omega*np.sqrt(sample_rate)

	def get_STD_static(self, n_points, sample_rate):
		'''
		Get the variance of the noise (combo of given static noise and of the spectrum, 
                                 integration to =0.1Hz) --> ~10sec sample of the noise
		'''
		# max formula :)
		static_noise_of_spectrum_function = 0
		n_points = 2*2**(int(np.log2(n_points))+1)
		if self.spectrum is not None:
			high_freq_cutoff = sample_rate/n_points
# 			print("low_freq_cutoff: ",self.low_freq_cutoff)
# 			print("freq_lower_bound 2: ",freq_lower_bound)
			static_noise_of_spectrum_function = 1./np.pi*(quad(self.spectrum, 
                                                      self.low_freq_cutoff*2.*np.pi, 
                                                      high_freq_cutoff*2.*np.pi,
                                                      points= np.linspace(2.*np.pi,
                                                                          high_freq_cutoff*np.pi,10))[0])
# 		print("noise: ",[np.sqrt(self.STD_SQUARED),
#                     np.sqrt(static_noise_of_spectrum_function),
#                     np.sqrt(self.STD_SQUARED) + np.sqrt(static_noise_of_spectrum_function)])
		return np.sqrt(self.STD_SQUARED) + np.sqrt(static_noise_of_spectrum_function)

@dataclass
class hamiltonian_data():
	matrix : np.ndarray
	pulse_data : pulse
	signal_type : int
	dsp : int = 0
	noise : noise_desciption = noise_desciption()

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
				if other.pulse_data != None:
					if hamiltonian.pulse_data != None:
						hamiltonian.pulse_data += other.pulse_data
					else:
						hamiltonian.pulse_data = other.pulse_data
				hamiltonian.noise += other.noise

				# other added.
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