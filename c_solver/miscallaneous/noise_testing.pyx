from cyarma_lib.cyarma cimport Col, vec2np, np2vec

cimport numpy as np
import numpy as np

cdef extern from "noise_functions.h":
	struct noise_specifier:
		int noise_type
		Col[double] STD_omega
		double STD_static

	Col[double] py_get_noise_from_spectral_density(Col[double] STD_omega, int n_samples);


def return_noise(std, n_samples):
	cdef np.ndarray[ double, ndim=1 ] STD_omega_np = std
	cdef Col[double] my_noise
	my_noise = py_get_noise_from_spectral_density(np2vec(STD_omega_np), n_samples)

	cdef np.ndarray[ double, ndim=1 ] noise_np = np.zeros(n_samples)
	vec2np(my_noise, noise_np)
	return noise_np
