#include <chrono> 
#include <armadillo>

struct noise_specifier{
	int type;
	arma::vec noise_spectral_density;
	double T2;
	double noise_power;
};

arma::vec get_white_noise(int steps);
arma::vec get_gaussian_noise(double T2, int steps);
arma::vec get_noise_from_spectral_density(arma::vec* noise_spectrum, double noise_power, int n_samples);