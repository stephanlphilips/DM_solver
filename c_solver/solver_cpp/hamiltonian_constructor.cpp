#include "hamiltonian_constructor.h"
#include "noise_functions.h"
#include "definitions.h"


hamiltonian_constructor::hamiltonian_constructor(int n_elem, int size, double delta_time, std::vector<data_object>* hamiltonian_data_objects){
	H_static = arma::cx_cube(arma::zeros<arma::cube>(size,size,n_elem),arma::zeros<arma::cube>(size,size,n_elem));
	H_FULL = arma::cx_cube(arma::zeros<arma::cube>(size,size,n_elem),arma::zeros<arma::cube>(size,size,n_elem));
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

		if (H_data_object->hamiltonian_type == EXP_H and H_data_object->noise_specs.noise_type == NO_NOISE){
			for (uint i = 0; i < H_static.n_slices; ++i){
				H_static.slice(i) += H_data_object->input_matrix * exp(H_data_object->input_vector.at(i));
			}
		}
	}
}

arma::cx_cube* hamiltonian_constructor::load_full_hamiltonian(){
	/*
	TODO load in noise here!
	add RWA part!

	Note, no check of existance of memory is done here. assumed to be fine.
	*/
	H_FULL = H_static;
	

	arma::cx_mat static_noise_H;
	arma::vec noise_vector = arma::zeros<arma::vec>(H_static.n_slices);

	for (std::vector<data_object>::iterator H_data_object = hamiltonian_data->begin(); H_data_object != hamiltonian_data->end(); ++H_data_object)
	{
		if (H_data_object->noise_specs.noise_type != NO_NOISE){
			if (H_data_object->noise_specs.noise_type == STATIC_NOISE or H_data_object->noise_specs.noise_type == STATIC_NOISE + SPECTRUM_NOISE){
				noise_vector += get_gaussian_noise(H_data_object->noise_specs.STD_static);
			}
			if (H_data_object->noise_specs.noise_type == SPECTRUM_NOISE or H_data_object->noise_specs.noise_type == STATIC_NOISE + SPECTRUM_NOISE){
				noise_vector += get_noise_from_spectral_density(&H_data_object->noise_specs.STD_omega, H_static.n_slices);
			}

			if (H_data_object->hamiltonian_type == EXP_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
					H_FULL.slice(i) += H_data_object->input_matrix *
							exp(H_data_object->input_vector.at(i) + noise_vector.at(i));
				}
			}else{
				for (uint i = 0; i < H_static.n_slices; ++i){
					H_FULL.slice(i) += noise_vector.at(i) * H_data_object->input_matrix;
				}
			}

		}

	}

	H_FULL *= delta_t;

	return &H_FULL;
}