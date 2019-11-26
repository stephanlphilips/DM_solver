#ifndef NOISE_FUNCTIONS_H
#define NOISE_FUNCTIONS_H

#include <chrono> 
#include <armadillo>

struct noise_specifier{
	int noise_type;
	double STD_static;
	arma::vec STD_omega;
};

arma::vec get_white_noise(int steps);
arma::vec get_gaussian_noise(double STD_static, int steps);
double get_gaussian_noise(double STD_static);
arma::vec get_noise_from_spectral_density(arma::vec* STD_omega, int n_samples);

#endif