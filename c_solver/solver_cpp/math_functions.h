#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

#include <armadillo>
#include <cmath>

arma::cx_mat custom_matrix_exp(arma::cx_mat input_matrix);
arma::cx_mat matrix_exp_Hamiltonian(arma::cx_mat H);

#endif