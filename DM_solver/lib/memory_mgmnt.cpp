#include "memory_mgmnt.h"
#include <cmath>
data_manager::data_manager(arma::cx_mat psi0, int n_elem, int size, int batch_size_input){
	n_steps = std::ceil(n_elem/(double )batch_size_input);
	calc_distro = arma::linspace<arma::Col<int> >(0, n_elem, n_steps+1);

	DM = arma::cx_cube(arma::zeros<arma::cube>(size,size,n_elem+1),arma::zeros<arma::cube>(size,size,n_elem+1));
	DM.slice(0) = psi0;

	U = arma::cx_cube(arma::zeros<arma::cube>(size,size,n_elem),arma::zeros<arma::cube>(size,size,n_elem));
	U_SliceCompleted = arma::cx_cube(arma::zeros<arma::cube>(size,size,n_steps),arma::zeros<arma::cube>(size,size,n_steps));
	U_final = arma::cx_mat(arma::zeros<arma::mat>(size,size),arma::zeros<arma::mat>(size,size));
};
