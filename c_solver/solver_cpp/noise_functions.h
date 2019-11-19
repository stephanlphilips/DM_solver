#ifndef NOISE_FUNCTIONS_H
#define NOISE_FUNCTIONS_H

#include <chrono> 
#include <armadillo>

struct noise_specifier{
	int noise_type;
	double T2;
	double noise_power;
	arma::vec S_omega_sqrt;
};

arma::vec get_white_noise(int steps);
arma::vec get_gaussian_noise(double T2, int steps);
double get_gaussian_noise(double T2);
arma::vec get_noise_from_spectral_density(arma::vec* noise_spectrum, double noise_power, int n_samples);

#endif