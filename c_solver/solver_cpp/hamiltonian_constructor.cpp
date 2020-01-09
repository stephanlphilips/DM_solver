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
		
		if (H_data_object->hamiltonian_type == RWA_H){
			for (int i = 0; i < n_elem; ++i){
				H_static.slice(i) += trimatu(H_data_object->input_matrix) * H_data_object->input_vector.at(i);
				H_static.slice(i) += trimatl(H_data_object->input_matrix, -1) * std::conj(H_data_object->input_vector.at(i));
			}
		}
		
		//std::cout<< H_data_object->hamiltonian_type << "\n";

		if (H_data_object->hamiltonian_type == EXP_H and H_data_object->noise_specs.noise_type == NO_NOISE){
			for (uint i = 0; i < H_static.n_slices; ++i){
				H_static.slice(i) += H_data_object->input_matrix * exp(2.0*(H_data_object->input_vector.at(i)));
				//std::cout<< exp(2.0*(H_data_object->input_vector.at(i))) << "\n";
			}
		}
		
		if (H_data_object->hamiltonian_type == EXPSAT_H and H_data_object->noise_specs.noise_type == NO_NOISE){
			for (uint i = 0; i < H_static.n_slices; ++i){
				H_static.slice(i) += H_data_object->input_matrix * pow(sqrt(1.0+exp(-2.0*(H_data_object->input_vector.at(i) +1.0/sqrt(2.0)))) - exp(-1.0*(H_data_object->input_vector.at(i) +1.0/sqrt(2.0))),2.0);
			}
		}
	}
}

arma::cx_cube* hamiltonian_constructor::load_full_hamiltonian(){
	H_FULL = H_static;

	arma::cx_mat static_noise_H;
	arma::vec noise_vector = arma::zeros<arma::vec>(H_static.n_slices);

	for (std::vector<data_object>::iterator H_data_object = hamiltonian_data->begin(); H_data_object != hamiltonian_data->end(); ++H_data_object)
	{
		if (H_data_object->noise_specs.noise_type != NO_NOISE){
			//std::cout<< "line 46 " << H_data_object->hamiltonian_type << " from " << NORM_H << "\n";
			if (H_data_object->noise_specs.noise_type == STATIC_NOISE or H_data_object->noise_specs.noise_type == STATIC_NOISE + SPECTRUM_NOISE){
				noise_vector += get_gaussian_noise(H_data_object->noise_specs.STD_static);
			}
			//std::cout<< "line 50 "<< H_data_object->hamiltonian_type << " from " << EXP_H << "\n";
			if (H_data_object->noise_specs.noise_type == SPECTRUM_NOISE or H_data_object->noise_specs.noise_type == STATIC_NOISE + SPECTRUM_NOISE){
				noise_vector += get_noise_from_spectral_density(&H_data_object->noise_specs.STD_omega, H_static.n_slices);
			}
			//std::cout<< "line 54 "<< H_data_object->hamiltonian_type << " from " << EXPSAT_H << "\n";
			if (H_data_object->hamiltonian_type == EXP_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
					H_FULL.slice(i) += H_data_object->input_matrix * exp(2.0*(H_data_object->input_vector.at(i) + noise_vector.at(i)));
					//std::cout<<"line58 vector0 " << H_data_object->input_vector.at(i) << "," << "vector1" <<  noise_vector.at(i) << "\n";
					//std::cout<<"line59 result " << exp(2.0*(H_data_object->input_vector.at(i) + noise_vector.at(i))) << "\n";
				}
			}
			if (H_data_object->hamiltonian_type == EXPSAT_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
					H_FULL.slice(i) += H_data_object->input_matrix * pow(sqrt(1.0+exp(-2.0*(H_data_object->input_vector.at(i) + noise_vector.at(i)+1.0/sqrt(2.0)))) - exp(-1.0*(H_data_object->input_vector.at(i) + noise_vector.at(i)+1.0/sqrt(2.0))),2.0);
					//std::cout<<"vector1" << H_data_object->input_vector.at(i) << "," <<"vector1" <<  noise_vector.at(i) << "\n";
					//std::cout<< pow(sqrt(1.0+exp(-2.0*(H_data_object->input_vector.at(i) + noise_vector.at(i)+1.0/sqrt(2.0)))) - exp(-1.0*(H_data_object->input_vector.at(i) + noise_vector.at(i)+1.0/sqrt(2.0))),2.0) << "\n";
				}
			}
			else{
				if (H_data_object->hamiltonian_type == RWA_H){
					for (uint i = 0; i < H_static.n_slices; ++i){
						H_FULL.slice(i) += trimatu(H_data_object->input_matrix) * noise_vector.at(i);
						H_FULL.slice(i) += trimatl(H_data_object->input_matrix, -1) * std::conj(noise_vector.at(i));
					}
				}
				else{
					for (uint i = 0; i < H_static.n_slices; ++i){
						H_FULL.slice(i) += noise_vector.at(i) * H_data_object->input_matrix;
					//std::cout<< (noise_vector.at(i)) << "\n";
					}
				}
			}

		}

	}

	H_FULL *= delta_t;

	return &H_FULL;
}