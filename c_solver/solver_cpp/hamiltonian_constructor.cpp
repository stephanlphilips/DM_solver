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
// 				std::cout<<"H0"<< H_data_object->input_matrix/ ( 1. - pow(H_data_object->input_vector.at(i) ,2.0)) << "\n";
			}
		}
		
		if (H_data_object->hamiltonian_type == RWA_H){
			for (int i = 0; i < n_elem; ++i){
				H_static.slice(i) += trimatu(H_data_object->input_matrix) * H_data_object->input_vector.at(i);
				H_static.slice(i) += trimatl(H_data_object->input_matrix) * std::conj(H_data_object->input_vector.at(i));
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
				H_static.slice(i) += H_data_object->input_matrix * pow(sqrt(1.0+exp(-2.0*(H_data_object->input_vector.at(i) +log(2.0)))) - exp(-1.0*(H_data_object->input_vector.at(i) +log(2.0))),2.0);
			}
		}
		
		
		if (H_data_object->hamiltonian_type == TANH_H and H_data_object->noise_specs.noise_type == NO_NOISE){
			for (uint i = 0; i < H_static.n_slices; ++i){
				H_static.slice(i) += H_data_object->input_matrix * tanh(H_data_object->input_vector.at(i));
			}
		}
		
		
		if (H_data_object->hamiltonian_type == SWO1_H and H_data_object->noise_specs.noise_type == NO_NOISE){
			for (uint i = 0; i < H_static.n_slices; ++i){
				H_static.slice(i) += H_data_object->input_matrix / ( 1. - pow(H_data_object->input_vector.at(i) ,2.0));
// 				std::cout<< "H1_heis" << H_data_object->input_matrix/ ( 1. - pow(H_data_object->input_vector.at(i) ,2.0)) << "\n";
			}
		}
		
		if (H_data_object->hamiltonian_type == SWO2_H and H_data_object->noise_specs.noise_type == NO_NOISE){
			for (uint i = 0; i < H_static.n_slices; ++i){
				H_static.slice(i) += H_data_object->input_matrix * ( 1. + pow(H_data_object->input_vector.at(i) ,2.0)) / pow( 1. - pow(H_data_object->input_vector.at(i) ,2.0),3.0);
			}
		}
	}
}

arma::cx_cube* hamiltonian_constructor::load_full_hamiltonian(){
	H_FULL = H_static;
// 	std::cout<< "correlated noise is turned off" << " \n";
	for (std::vector<data_object>::iterator H_data_object = hamiltonian_data->begin(); H_data_object != hamiltonian_data->end(); ++H_data_object)
	{
		arma::vec noise_vector = arma::zeros<arma::vec>(H_static.n_slices);
		if (H_data_object->noise_specs.noise_type != NO_NOISE){
			if (H_data_object->noise_specs.noise_type == STATIC_NOISE or H_data_object->noise_specs.noise_type == STATIC_NOISE + SPECTRUM_NOISE){
				noise_vector += get_gaussian_noise(H_data_object->noise_specs.STD_static);
			}
			if (H_data_object->noise_specs.noise_type == SPECTRUM_NOISE or H_data_object->noise_specs.noise_type == STATIC_NOISE + SPECTRUM_NOISE){
				noise_vector += get_noise_from_spectral_density(&H_data_object->noise_specs.STD_omega, H_static.n_slices);
			}
			if (H_data_object->hamiltonian_type == EXP_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
					H_FULL.slice(i) += H_data_object->input_matrix * exp(2.0*(H_data_object->input_vector.at(i) + noise_vector.at(i)));
				}
			}
			if (H_data_object->hamiltonian_type == EXPSAT_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
					H_FULL.slice(i) += H_data_object->input_matrix * pow(sqrt(1.0+exp(-2.0*(H_data_object->input_vector.at(i) + noise_vector.at(i)+log(2.0)))) - exp(-1.0*(H_data_object->input_vector.at(i) + noise_vector.at(i)+log(2.0))),2.0);
				}
			}
			
			
			if (H_data_object->hamiltonian_type == TANH_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
					H_FULL.slice(i) += H_data_object->input_matrix * tanh(H_data_object->input_vector.at(i) + noise_vector.at(i));
				}
			}
			
			
			if (H_data_object->hamiltonian_type == SWO1_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
					H_FULL.slice(i) += H_data_object->input_matrix / (1. - pow(H_data_object->input_vector.at(i) + noise_vector.at(i),2.0));
				}
			}
			
			if (H_data_object->hamiltonian_type == SWO2_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
					H_FULL.slice(i) += H_data_object->input_matrix * ( 1. + pow(H_data_object->input_vector.at(i) + noise_vector.at(i) ,2.0))  / pow(1. - pow(H_data_object->input_vector.at(i) + noise_vector.at(i),2.0),3.0);
				}
			}
			
			
			
			if (H_data_object->hamiltonian_type == RWA_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
						H_FULL.slice(i) += trimatu(H_data_object->input_matrix) * noise_vector.at(i);
						H_FULL.slice(i) += trimatl(H_data_object->input_matrix) * std::conj(noise_vector.at(i));
				}
			}
			if (H_data_object->hamiltonian_type == NORM_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
						H_FULL.slice(i) += noise_vector.at(i) * H_data_object->input_matrix;
					
					
				}
// 				std::cout<< H_data_object->input_matrix << "\n";
// 				std::cout<< "noise is " << (noise_vector.at(0))  << "\n";
// 				std::cout<< (noise_vector.at(0)) << "\n";
			}
			

		}

	}
	//std::cout<<"exchange End" << H_FULL.slice(0).at(1,2) * 0.15915494309189535 << "\n";
	H_FULL *= delta_t;
	
	return &H_FULL;
}

arma::cx_cube* hamiltonian_constructor::load_full_hamiltonian_correlated_noise(arma::mat correlation_matrix_static, arma::mat correlation_matrix_dynamic){
	H_FULL = H_static;
// 	std::cout<< "correlated noise is turned on: " << correlation_matrix_static.n_cols << " \n";
	//uint slices_static = correlation_matrix_static.n_cols;
	//uint slices_dynamic = correlation_matrix_dynamic.n_cols;
	arma::mat noise_matrix_static = arma::zeros<arma::mat>(H_static.n_slices,correlation_matrix_static.n_cols);
	arma::mat noise_matrix_dynamic= arma::zeros<arma::mat>(H_static.n_slices,correlation_matrix_dynamic.n_cols);
// 	std::cout<< correlation_matrix_static<< " \n";
	uint counter_static = 0;
	uint counter_dynamic = 0;
// 	std::cout<<"number of columns static"<< noise_matrix_static.n_cols << " \n";
// 	std::cout<<"number of columns dynamic"<< correlation_matrix_dynamic.n_cols << " \n";
	for (std::vector<data_object>::iterator H_data_object = hamiltonian_data->begin(); H_data_object != hamiltonian_data->end(); ++H_data_object)
	{
	
		if (H_data_object->noise_specs.noise_type == STATIC_NOISE){
			arma::vec noise_vector = arma::zeros<arma::vec>(H_static.n_slices);
			if (H_data_object->noise_specs.noise_type == STATIC_NOISE or H_data_object->noise_specs.noise_type == STATIC_NOISE + SPECTRUM_NOISE){
				noise_vector += get_gaussian_noise(H_data_object->noise_specs.STD_static);
			}
			if (H_data_object->noise_specs.noise_type == SPECTRUM_NOISE or H_data_object->noise_specs.noise_type == STATIC_NOISE + SPECTRUM_NOISE){
				noise_vector += get_noise_from_spectral_density(&H_data_object->noise_specs.STD_omega, H_static.n_slices);
			}
// 			std::cout<< "marker line 161: " << counter_static << " \n";
// 			std::cout<< noise_matrix_static<< " \n";
			noise_matrix_static.col(counter_static)+= noise_vector;
			counter_static++;
// 			std::cout<<"static"<< counter_static << " \n";
		}
		
		if (H_data_object->noise_specs.noise_type == SPECTRUM_NOISE or H_data_object->noise_specs.noise_type == STATIC_NOISE + SPECTRUM_NOISE){
			arma::vec noise_vector = arma::zeros<arma::vec>(H_static.n_slices);
			if (H_data_object->noise_specs.noise_type == STATIC_NOISE or H_data_object->noise_specs.noise_type == STATIC_NOISE + SPECTRUM_NOISE){
				noise_vector += get_gaussian_noise(H_data_object->noise_specs.STD_static);
			}
			if (H_data_object->noise_specs.noise_type == SPECTRUM_NOISE or H_data_object->noise_specs.noise_type == STATIC_NOISE + SPECTRUM_NOISE){
				noise_vector += get_noise_from_spectral_density(&H_data_object->noise_specs.STD_omega, H_static.n_slices);
			}
//  			std::cout<< noise_matrix_dynamic.col(counter_dynamic)<< " \n";
			noise_matrix_dynamic.col(counter_dynamic)+= noise_vector;
			counter_dynamic++;
// 			std::cout<<"dynamic"<< counter_dynamic << " \n";
		}
	}
//  	std::cout<<"max dynamic"<< counter_static << " \n";
//  	std::cout<<"max static"<< counter_dynamic << " \n";
	counter_static = 0;
	counter_dynamic = 0;
	
	for (std::vector<data_object>::iterator H_data_object = hamiltonian_data->begin(); H_data_object != hamiltonian_data->end(); ++H_data_object)
	{
		arma::vec noise_vector = arma::zeros<arma::vec>(H_static.n_slices);
		if (H_data_object->noise_specs.noise_type != NO_NOISE){
		
			if (H_data_object->noise_specs.noise_type == STATIC_NOISE){
				for (uint it = 0; it < correlation_matrix_static.n_cols; ++it){
					noise_vector += noise_matrix_static.col(it)*correlation_matrix_static(counter_static,it);
				}
			
			counter_static++;
// 			std::cout<<"static in"<< counter_static << " \n";
			}
			if (H_data_object->noise_specs.noise_type == SPECTRUM_NOISE or H_data_object->noise_specs.noise_type == STATIC_NOISE + SPECTRUM_NOISE){
				for (uint it = 0; it < correlation_matrix_dynamic.n_cols; ++it){
					noise_vector += noise_matrix_dynamic.col(it)*correlation_matrix_dynamic(counter_dynamic,it);
				}
			
			counter_dynamic++;
// 			std::cout<<"dynamic in"<< counter_dynamic << " \n";
			}
			if (H_data_object->hamiltonian_type == EXP_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
					H_FULL.slice(i) += H_data_object->input_matrix * exp(2.0*(H_data_object->input_vector.at(i) + noise_vector.at(i)));
				}
			}
			if (H_data_object->hamiltonian_type == EXPSAT_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
					H_FULL.slice(i) += H_data_object->input_matrix * pow(sqrt(1.0+exp(-2.0*(H_data_object->input_vector.at(i) + noise_vector.at(i)+log(2.0)))) - exp(-1.0*(H_data_object->input_vector.at(i) + noise_vector.at(i)+log(2.0))),2.0);
				}
			}
			
			
			if (H_data_object->hamiltonian_type == TANH_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
					H_FULL.slice(i) += H_data_object->input_matrix * tanh(H_data_object->input_vector.at(i) + noise_vector.at(i));
				}
			}
			
			
			if (H_data_object->hamiltonian_type == SWO1_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
					H_FULL.slice(i) += H_data_object->input_matrix / (1. - pow(H_data_object->input_vector.at(i) + noise_vector.at(i),2.0));
				}
			}
			
			if (H_data_object->hamiltonian_type == SWO2_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
					H_FULL.slice(i) += H_data_object->input_matrix * ( 1. + pow(H_data_object->input_vector.at(i) + noise_vector.at(i) ,2.0))  / pow(1. - pow(H_data_object->input_vector.at(i) + noise_vector.at(i),2.0),3.0);
				}
			}
			
			
			
			if (H_data_object->hamiltonian_type == RWA_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
						H_FULL.slice(i) += trimatu(H_data_object->input_matrix) * noise_vector.at(i);
						H_FULL.slice(i) += trimatl(H_data_object->input_matrix) * std::conj(noise_vector.at(i));
				}
			}
			if (H_data_object->hamiltonian_type == NORM_H){
				for (uint i = 0; i < H_static.n_slices; ++i){
						H_FULL.slice(i) += noise_vector.at(i) * H_data_object->input_matrix;
					
					
				}
// 				std::cout<< H_data_object->input_matrix << "\n";
// 				std::cout<< "noise is " << (noise_vector.at(0))  << "\n";
// 				std::cout<< (noise_vector.at(0)) << "\n";
			}
			

		}

	}
// 	std::cout<<"upper diagonal" << H_FULL.slice(0).at(0,0) * 0.15915494309189535 << "\n";
// 	std::cout<<"lower diagonal" << H_FULL.slice(0).at(1,1) * 0.15915494309189535 << "\n";
//	std::cout<<"H_FULL" << H_FULL.slice(0) * 0.15915494309189535 << "\n";
	H_FULL *= delta_t;
	
	return &H_FULL;
}
