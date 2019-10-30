#include "hamiltonian_constructor.h"
#include "definitions.h"


hamiltonian_constructor::hamiltonian_constructor(int n_elem, int size, double delta_time, std::vector<data_object>* hamiltonian_data_objects){
	H_static = arma::cx_cube(arma::zeros<arma::cube>(size,size,n_elem),arma::zeros<arma::cube>(size,size,n_elem));
	H_FULL =arma::cx_cube(arma::zeros<arma::cube>(size,size,n_elem),arma::zeros<arma::cube>(size,size,n_elem));
	delta_t = delta_time;
	hamiltonian_data = hamiltonian_data_objects;

	// fill up static part of the hamiltonian
	for (std::vector<data_object>::iterator H_data_object = hamiltonian_data->begin(); H_data_object != hamiltonian_data->end(); ++H_data_object)
	{
		if (H_data_object->hamiltonian_type == NORM_H){
			for (int i = 0; i < n_elem; ++i){
				H_static.slice(i) += H_data_object->input_matrix * H_data_object->input_vector.at(i);
			}
		}
		std::cout<< "H=contructor slice test :: \n" << H_data_object->input_matrix * H_data_object->input_vector.at(0) << "\n" << H_data_object->input_vector.at(0) << "\n" ;
	}
}

arma::cx_cube* hamiltonian_constructor::load_full_hamiltonian(){
	/*
	TODO load in noise here!
	add RWA part!

	Note, no check of existance of memory is done here. assumed to be fine.
	*/
	H_FULL = H_static;
	
	for (std::vector<data_object>::iterator H_data_object = hamiltonian_data->begin(); H_data_object != hamiltonian_data->end(); ++H_data_object)
	{
		if (H_data_object->hamiltonian_type == EXP_H){
			for (int i = 0; i < H_static.size(); ++i){
				H_FULL.slice(i) += H_data_object->input_matrix * exp(H_data_object->input_vector.at(i));
			}
		}
	}

	H_FULL *= delta_t;
	std::cout<< "H FULL =contructor slice test :: \n" << H_FULL.slice(0) << "\n" ;

	return &H_FULL;
}