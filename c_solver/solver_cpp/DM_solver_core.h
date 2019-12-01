#ifndef DM_SOLVER_CORE_H
#define DM_SOLVER_CORE_H

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
#include "noise_functions.h"

class DM_solver_calc_engine
{
	int size;
	int iterations;
	bool do_Lindblad;
	arma::cx_cube my_density_matrices;
	arma::cx_cube unitaries;
	std::vector<data_object> input_data;
	std::vector<lindblad_obj> lindblad_oper;
public:
	DM_solver_calc_engine(int size_matrix);
	void add_H1(arma::cx_mat input_matrix, arma::cx_vec time_dep_data, int hamiltonian_type, noise_specifier noise_specs);
	void add_lindbladian(arma::cx_mat A, double gamma);
	void set_number_of_evalutions(int iter);
	void calculate_evolution(arma::cx_mat psi0, double end_time, int steps);

	arma::mat return_expectation_values(arma::cx_cube input_matrices);
	arma::cx_mat get_unitaries();
	arma::cx_mat get_lastest_rho();
	arma::cx_cube get_all_density_matrices();
};

#endif