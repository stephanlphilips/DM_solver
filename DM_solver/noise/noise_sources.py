from scipy.integrate import quad

import numpy as np
import copy

class static_noise_generator():
	def __init__(self, sigma_noise):
		'''
		add static noise generated from a Gaussian distribution

		Args:
			sigma_noise (float) : standard deviation of the noise (sigma = sqrt(2)/T_2)
		'''
		self.amplitude = sigma_noise
		self.generator = np.random.default_rng()

	def render_noise(self, npt, sample_rate=None):
		return np.ones((npt,)) * self.generator.normal(scale = self.amplitude, size=1)

	def __mul__(self, other):
		new = copy.copy(self)
		new.amplitude *= other
		return new
	
	def __rmul__(self, other):
		return self.__mul__(other)

	def __copy__(self):
		noise_gen =  static_noise_generator(self.amplitude)
		noise_gen.generator = copy.copy(self.generator)
		return noise_gen


class spectral_noise_generator():
	def __init__(self, amplitude, spectrum, low_freq_cutoff=None):
		'''
		add noise that corresponds to a noise spectral density (S(f) = A*s(f))

		Args:
			amplitude (float) : amplitude of the noise spectral density function
			spectrum (function) : function that defines the spectrum (e.g. 1/f)
			low_freq_cutoff (float) : adds the static noise with an amplitude corresponding to the spectrum
									  (neglected when None) 
		'''
		self.spectrum  = lambda x: spectrum(x)*amplitude
		self.generator = np.random.default_rng()
		self.low_freq_cutoff = low_freq_cutoff

	def render_noise(self, npt, sample_rate):
		if self.low_freq_cutoff is None:
			return self.__render_dynamic_noise(npt, sample_rate)

		return self.__render_static_noise(npt, sample_rate) + self.__render_dynamic_noise(npt, sample_rate)

	def __render_dynamic_noise(self, npt, sample_rate):
		# more info : Timmer, J. and Koenig, M. On generating power law noise. Astron. Astrophys. 300, 707-710 (1995)
		n_points = 2*2**(int(np.log2(npt)))
		f = np.fft.rfftfreq(n_points, d=1/sample_rate)
		
		sigma_noise = lambda x: np.sqrt(self.spectrum(x))/2
		sigma_sampled = (self.generator.normal(size=f.size, scale=sigma_noise(f)) + 
						1j*self.generator.normal(size=f.size, scale=sigma_noise(f)))
		sigma_sampled = np.concatenate(([0], sigma_sampled[1:(n_points+1)//2], sigma_sampled[1:].conjugate()[::-1]))
		
		return np.real(np.fft.ifft(sigma_sampled, norm='ortho')[:npt])*np.sqrt(sample_rate)

	def __render_static_noise(self, npt, sample_rate):
		nyquist_freq = sample_rate/2
		sigma_noise = np.sqrt(2*quad(self.spectrum, 
									self.low_freq_cutoff*2*np.pi, nyquist_freq*2*np.pi))

		return np.ones((npt,)) * self.generator.normal(scale = sigma_noise, size=1)

	def __mul__(self, other):
		new = spectral_noise_generator(other**2, self.spectrum, self.low_freq_cutoff)
		new.generator = copy.copy(self.generator)
		return new
	
	def __rmul__(self, other):
		return self.__mul__(other)

	def __copy__(self):
		noise_gen =  spectral_noise_generator(1, self.spectrum, self.low_freq_cutoff)
		noise_gen.generator = copy.copy(self.generator)
		return noise_gen

if __name__ == '__main__':
	import scipy.signal
	import matplotlib.pyplot as plt
	# static_gen = static_noise_generator(5)
	# static_gen2 =  static_gen*2

	# plt.plot(static_gen.render_noise(100))
	# plt.plot(static_gen2.render_noise(100))
	# plt.show()
	
	amp       = 5
	npt = 50000
	sample_rate = 100

	sng1 = spectral_noise_generator(amp, lambda x: 1/x)
	sng2 = 2*sng1 #double the noise and make a copy
	sng3 = spectral_noise_generator(amp, lambda x: 1/x/x)

	# check if copies are correlated
	print('correlated generators')
	print(sng1.render_noise(npt=10, sample_rate=1))
	print(sng2.render_noise(npt=10, sample_rate=1))

	# check if the spectrum is correct

	f_1, Pxx_1 = scipy.signal.welch(sng1.render_noise(npt, sample_rate), fs=sample_rate)
	f_2, Pxx_2 = scipy.signal.welch(sng3.render_noise(npt, sample_rate), fs=sample_rate)
	plt.plot(f_1, Pxx_1)
	plt.plot(f_2, Pxx_2)
	plt.plot(f_1, amp/f_1)
	plt.plot(f_2, amp/f_2/f_2)
	plt.yscale('log')
	plt.xscale('log')
	plt.show()