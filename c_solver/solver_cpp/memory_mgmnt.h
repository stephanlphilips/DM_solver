#ifndef MEMORY_MANAGEMENT_H
#define MEMORY_MANAGEMENT_H

#include <armadillo>

// small mananger that manages all data used generated in the calculation.
class data_manager
{
public:
	arma::cx_cube my_density_matrices;
	arma::Col<int> calc_distro;
	arma::cx_cube my_density_matrices_tmp;
	arma::cx_cube unitaries_cache;
	arma::cx_cube unitaries_finished_slices;
	arma::cx_cube unitaries;
	int number_of_calc_steps;
	bool done;
	
	data_manager(int n_elem, int size, int iterations, int batch_size_input);
	void init_iteration(arma::cx_mat psi0);
	void finish_iteration();
};


#endif