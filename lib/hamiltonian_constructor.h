void preload_hamilonian(std::vector<data_object_VonNeumannSolver>* my_init_data, double start_time, double stop_time, int steps){
	/* 
	function that constructs all the necesary elements to prepare for the construction of the hamiltonian. 
	things to consider here would e.g. be the pulse response of a AWG ( you need to know what happened before, so you cannot just ask random parts of the pulse.
	
	Another reason for preloading is the fact that we will go from time base, to the number of points (in the multiprocessing part of the code).
	*/ 

	// AWG pulses and MW'ves
	#pragma omp parallel for
	for (int i = 0; i < my_init_data->size(); ++i){
		switch(my_init_data->at(i).type){
			case 2:{
				my_init_data->at(i).AWG_obj.preload(start_time,stop_time, steps);
				break;
			}
			case 3:{
				my_init_data->at(i).MW_obj.preload(start_time,stop_time,steps);
			}

		}
	}
}

void preload_noise(std::vector<noise> *my_noise_data, double start_time, double stop_time, int steps){
	// Seperate preloader, as it as to be loaded for each iteration.
	#pragma omp parallel for
	for (int j = 0; j < my_noise_data->size(); ++j){
		my_noise_data->at(j).preload(start_time,stop_time,steps);
	}
}

void contruct_hamiltonian(arma::cx_cube* hamiltonian, int start, int stop, 
	double start_time, double stop_time ,double delta_t,
	std::vector<data_object_VonNeumannSolver>* my_init_data,
	std::vector<maxtrix_elem_depen_VonNeumannSolver>* my_parameter_depence,
	std::vector<noise> *my_noise_data){

	int steps = stop-start;

	for (int i = 0; i < my_init_data->size(); ++i){
		switch(my_init_data->at(i).type){
			case 0:{
				// Constant matrix
				hamiltonian->each_slice() += my_init_data->at(i).input_matrix1*delta_t;
				break;
			}
			case 1:{
				// Points given by user for each time point
				for (int j = start; j < stop; ++j){
					hamiltonian->slice(j-start) += my_init_data->at(i).input_matrix1*my_init_data->at(i).input_vector[j];
				}
				break;
			}
			case 2:{
				// Microwave type signal
				my_init_data->at(i).AWG_obj.fetch_H(hamiltonian, start, stop);
				break;
			}
			case 3:{
				// AWG signal
				my_init_data->at(i).MW_obj.fetch_H(hamiltonian, start, stop);
			}

		}
	}
	// Add noise:
	for (int j = 0; j < my_noise_data->size(); ++j){
		my_noise_data->at(j).fetch_noise(hamiltonian, start, stop);
	}

	// add parameter dependencies.
	for (int i = 0; i < my_parameter_depence->size(); ++i){
		if (my_parameter_depence->at(i).type ==0)
			generate_parameter_dependent_matrices(hamiltonian, my_parameter_depence->at(i).input_matrix, my_parameter_depence->at(i).i,
				my_parameter_depence->at(i).j, my_parameter_depence->at(i).matrix_param, delta_t);
	}

	for (int i = 0; i < my_parameter_depence->size(); ++i){
		if (my_parameter_depence->at(i).type ==1)
			generate_time_dependent_matrices(hamiltonian, my_parameter_depence->at(i).locations, my_parameter_depence->at(i).frequency, 
				start_time, stop_time, steps);
	}

}