#include <armadillo>
#include <iostream>
#include <cmath>
#include <complex>
#include <math.h>

#include <string>
#include <memory>
#include <map>

#include "hamiltonian_constructor.h"
#include "memory_mgmnt.h" 
#include "math_functions.h" 

class DM_solver_calc_engine
{
	int size;
	int iterations;
	arma::cx_cube my_density_matrices;
	arma::cx_mat unitary;
	std::vector<data_object> input_data;
public:
	DM_solver_calc_engine(int size_matrix);
	void add_H1(arma::cx_mat input_matrix, arma::cx_vec time_dep_data, int hamiltonian_type);
	void set_number_of_evalutions(int iter);
	void calculate_evolution(arma::cx_mat psi0, double end_time, int steps);

	arma::mat return_expectation_values(arma::cx_cube input_matrices);
	arma::cx_mat get_unitary();
	arma::cx_mat get_lastest_rho();
	arma::cx_cube get_all_density_matrices();
};
