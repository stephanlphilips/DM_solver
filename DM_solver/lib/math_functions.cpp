#include "math_functions.h"
#include <cmath>

arma::cx_mat matrix_exp_Hamiltonian(arma::cx_mat H){
	arma::vec eig_val;
	arma::cx_mat eig_vec;
	const std::complex<double> j(0, 1);

	arma::eig_sym(eig_val, eig_vec, H);
	return eig_vec*arma::diagmat(arma::exp(-j*eig_val))*eig_vec.t();
}