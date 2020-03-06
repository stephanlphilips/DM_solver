#include "math_functions.h"

arma::cx_mat custom_matrix_exp(arma::cx_mat input_matrix){
	// **depricated
	// ok, but can be more efficient using the pade method.
	// uses now matrix scaling in combination with a taylor

	// assume to be the same over the whole simulation 
	const int accuracy = 5;
    const int scaling = 5;
    const uint size = input_matrix.n_rows;
	const double scaling_factor = 1/std::pow(2., double(scaling));
    
    arma::cx_mat tmp = arma::cx_mat(arma::eye<arma::mat>(size,size),arma::zeros<arma::mat>(size,size));
	arma::cx_mat output_matrix = arma::cx_mat(arma::eye<arma::mat>(size,size),arma::zeros<arma::mat>(size,size));

    double factorial_i = 1.0;

    input_matrix *= scaling_factor;

	for(int i = 1; i < accuracy; i++) {
		factorial_i = factorial_i * i;
		tmp *= input_matrix;
		output_matrix += tmp/factorial_i;
	}

	for(int i=0; i < scaling; ++i)
		output_matrix *= output_matrix;

	return output_matrix;
} 

arma::cx_mat matrix_exp_Hamiltonian(arma::cx_mat H){
	// way more effecient than the other one.
	arma::vec eig_val;
	arma::cx_mat eig_vec;
	const std::complex<double> j(0, 1);
	arma::eig_sym(eig_val, eig_vec, H);
	
	return eig_vec*arma::diagmat(arma::exp(-j*eig_val))*eig_vec.t();
}