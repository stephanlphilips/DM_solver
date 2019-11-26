from cyarma_lib.cyarma cimport Col, vec2np, np2vec

cdef extern from "noise_functions.h":
	struct noise_specifier:
		int noise_type
		Col[double] STD_omega
		double STD_static

	Col[double] get_noise_from_spectral_density(Col[double]* STD_omega, int n_samples);


def return_noise(std, n_samples):
	cdef np.ndarray[ double, ndim=1 ] STD_omega_np = std
	return vec2np(get_noise_from_spectral_density(np2vec(STD_omega_np), n_samples))
