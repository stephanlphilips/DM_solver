#ifndef HAMILTONIAN_CONSTRUCTOR_H
#define HAMILTONIAN_CONSTRUCTOR_H

#include <armadillo>
#include <math.h> 
#include "noise_functions.h"

struct data_object
{
	int hamiltonian_type;
	//double leverarm;
	//double voltage_offset;
	arma::cx_mat input_matrix;
	arma::cx_vec input_vector;
	noise_specifier noise_specs;
};

class hamiltonian_constructor
{
	double delta_t;
	arma::cx_cube H_static;
	arma::cx_cube H_FULL;
	std::vector<data_object>* hamiltonian_data;
public:
	hamiltonian_constructor(int n_elem, int size, double delta_time, std::vector<data_object>* hamiltonian_data_object);
	arma::cx_cube* load_full_hamiltonian();
};

#endif