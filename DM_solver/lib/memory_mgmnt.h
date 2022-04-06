#ifndef MEMORY_MANAGEMENT_H
#define MEMORY_MANAGEMENT_H

#include <armadillo>

class data_manager
{
public:
	arma::Col<int> calc_distro;
	arma::cx_cube DM;
	arma::cx_cube U;
	arma::cx_cube U_SliceCompleted;
	arma::cx_mat  U_final;
	int n_steps;
	
	data_manager(arma::cx_mat psi0, int n_elem, int size, int batch_size_input);
};


#endif