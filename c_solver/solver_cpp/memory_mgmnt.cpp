#include "memory_mgmnt.h"

data_manager::data_manager(int n_elem, int size, int iterations, int batch_size_input){
	number_of_calc_steps = std::ceil(n_elem/batch_size_input);
	calc_distro = arma::linspace<arma::Col<int> >(0, n_elem, number_of_calc_steps+1);

	my_density_matrices = arma::cx_cube(arma::zeros<arma::cube>(size,size,n_elem+1),arma::zeros<arma::cube>(size,size,n_elem+1));
	my_density_matrices_tmp = arma::cx_cube(arma::zeros<arma::cube>(size,size,n_elem+1),arma::zeros<arma::cube>(size,size,n_elem+1));
	
	unitaries_cache = arma::cx_cube(arma::zeros<arma::cube>(size,size,n_elem),arma::zeros<arma::cube>(size,size,n_elem));
	unitaries_finished_slices = arma::cx_cube(arma::zeros<arma::cube>(size,size,number_of_calc_steps),arma::zeros<arma::cube>(size,size,number_of_calc_steps));
	
	unitaries = arma::cx_cube(arma::zeros<arma::cube>(size,size,iterations),arma::zeros<arma::cube>(size,size,iterations));

};

void data_manager::init_iteration(arma::cx_mat psi0){
	my_density_matrices_tmp.slice(0) = psi0;
	done = false;
};
void data_manager::finish_iteration(){
	my_density_matrices += my_density_matrices_tmp;
};