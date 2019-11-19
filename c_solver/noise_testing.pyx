from cyarma_lib.cyarma cimport Col, vec2np

cdef extern from "noise_functions.h":
	struct noise_specifier:
		int noise_type
		Col[double] noise_spectral_density;
		double T2
		double noise_power

	Col[double] get_white_noise(int steps);
	Col[double] get_gaussian_noise(double T2, int steps);
	Col[double] get_noise_from_spectral_density(Col[double]* noise_spectrum, double noise_power, int n_samples);


def get_white(int steps):
	noise = vec2np(get_white(steps))
	return noise

def gauss_noise(double T2, int steps):
	noise = get_gaussian_noise(T2, steps)
	return noise
