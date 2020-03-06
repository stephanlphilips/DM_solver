#ifndef NOISE_FUNCTIONS_H
#define NOISE_FUNCTIONS_H

#include <chrono> 
#include <armadillo>

struct noise_specifier{
	int noise_type;
	double STD_static;
	arma::vec STD_omega;
};

struct lindblad_obj{
	arma::cx_mat C; // C is \gamma*A
	arma::cx_mat C_dag;
	arma::cx_mat C_dag_C; // premultiplied matrices
	arma::cx_mat C_C_dag; // premultiplied matrices
};

double get_gaussian_noise(double STD_static);
arma::vec get_noise_from_spectral_density(arma::vec* STD_omega, int n_samples);
// remove later -- function for test purposes only
arma::vec py_get_noise_from_spectral_density(arma::vec STD_omega, int n_samples);


#endif