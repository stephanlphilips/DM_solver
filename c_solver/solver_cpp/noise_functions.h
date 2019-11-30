#ifndef NOISE_FUNCTIONS_H
#define NOISE_FUNCTIONS_H

#include <chrono> 
#include <armadillo>

struct noise_specifier{
	int noise_type;
	double STD_static;
	arma::vec STD_omega;
};

double get_gaussian_noise(double STD_static);
arma::vec get_noise_from_spectral_density(arma::vec* STD_omega, int n_samples);
// remove later -- function for test purposes only
arma::vec py_get_noise_from_spectral_density(arma::vec STD_omega, int n_samples);


#endif