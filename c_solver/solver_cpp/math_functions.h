#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

#include <armadillo>
#include <cmath>

std::complex<double> poly_function(std::complex<double> x, arma::cx_mat pertubation_function);
arma::cx_mat custom_matrix_exp(arma::cx_mat input_matrix);
arma::cx_vec integrate_cexp(double start, double stop, int steps, double frequency);
arma::vec fft_freq(int n, double f);

#endif