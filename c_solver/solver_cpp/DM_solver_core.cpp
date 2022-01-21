#include "DM_solver_core.h"


DM_solver_calc_engine::DM_solver_calc_engine(int size_matrix){
	size = size_matrix;
	iterations = 1;
	do_Lindblad = false;
	add_correlation = false;
	arma::mat correlation_matrix_static;
	arma::mat correlation_matrix_dynamic;
}
void DM_solver_calc_engine::add_H1(arma::cx_mat input_matrix, arma::cx_vec time_dep_data, int hamiltonian_type, noise_specifier noise_specs){
	data_object temp;
	temp.hamiltonian_type = hamiltonian_type;
	temp.input_matrix = input_matrix;
	temp.input_vector = time_dep_data;
	temp.noise_specs = noise_specs;
	input_data.push_back(temp);
}

void DM_solver_calc_engine::add_lindbladian(arma::cx_mat A, double gamma){ // NOTE: gamma = amplitude^2
		do_Lindblad = true;

		lindblad_obj lind_tmp;
		lind_tmp.C = gamma*A;
		lind_tmp.C_dag = lind_tmp.C.t();
		lind_tmp.C_dag_C = lind_tmp.C_dag * lind_tmp.C;

		lindblad_oper.push_back(lind_tmp);
	}

void DM_solver_calc_engine::add_correlation_matrix(arma::mat f_correlation_matrix_static, arma::mat f_correlation_matrix_dynamic){
// 	std::cout<< "correlation matrix 0: "<< correlation_matrix_static << "\n";
	DM_solver_calc_engine::correlation_matrix_static = f_correlation_matrix_static;
	DM_solver_calc_engine::correlation_matrix_dynamic = f_correlation_matrix_dynamic;
// 	std::cout<< "correlation matrix 1: "<< correlation_matrix_static << "\n";
	add_correlation = true;
}

void DM_solver_calc_engine::set_number_of_evalutions(int iter){
	iterations = iter;
}


void DM_solver_calc_engine::calculate_evolution(arma::cx_mat psi0, double end_time, int steps){
	// per multithread steps, process 1000 unitaries
	int batch_size = 1000;
	//int iterator_loop
	//int batch_size = 1;
	std::cout<< "set batch size to "<< batch_size << "\n";
 	//arma::arma_version ver;
 	//std::cout << "ARMA version: " << ver.as_string() << std::endl;
 	
	double delta_t = end_time/steps;
    hamiltonian_constructor hamiltonian_mgr = hamiltonian_constructor(steps, size, delta_t, &input_data);
	data_manager data_mgr = data_manager(steps,size, iterations, batch_size);

	const std::complex<double> j(0, 1);
	std::complex<double> average_exchange; 
	std::complex<double> average_zeeman; 
	

	for (int iteration = 0; iteration < iterations; ++iteration){
		if ((iterations != 1) && (iteration % 50 == 0))
			std::cout<< "iterations " << iteration << "\n";
			
		arma::cx_cube* hamiltonian;
		data_mgr.init_iteration(psi0);
// 		std::cout<< "correlation matrix 2:  "<< correlation_matrix_static << "\n";
		if (!add_correlation){
			hamiltonian = hamiltonian_mgr.load_full_hamiltonian();
		}else{
			hamiltonian = hamiltonian_mgr.load_full_hamiltonian_correlated_noise(DM_solver_calc_engine::correlation_matrix_static,DM_solver_calc_engine::correlation_matrix_dynamic);
		}
// 		std::cout<< hamiltonian->slice(0)* 0.15915494309189535/delta_t<< "\n";
// 		std::cout<< "difference: " << (hamiltonian->slice(0).at(0,0)-hamiltonian->slice(0).at(1,1))* 0.15915494309189535/delta_t<< "\n";
// 		std::cout<< "difference: " << (hamiltonian->slice(1000).at(0,0)-hamiltonian->slice(1000).at(1,1))* 0.15915494309189535/delta_t<< "\n";
		//average_ham += arma::abs(hamiltonian->slice(0) * 0.15915494309189535/delta_t,2);
		average_zeeman += std::pow(std::abs(1.0*(hamiltonian->slice(0).at(1,1) - hamiltonian->slice(0).at(2,2))/delta_t * 0.15915494309189535),2);
		average_exchange += std::pow(std::abs(hamiltonian->slice(0).at(1,2) * 0.15915494309189535/delta_t),2);

		// calculate unitaries
		#pragma omp parallel shared(hamiltonian, data_mgr)
		{
			#pragma omp for
			for (int calc_step_number=0; calc_step_number < data_mgr.number_of_calc_steps; calc_step_number++){
				int init = data_mgr.calc_distro(calc_step_number);
				int end = data_mgr.calc_distro(calc_step_number+1);
// 				std::cout<< "calc_step_number " << calc_step_number << "\n";
                
                
				data_mgr.unitaries_finished_slices.slice(calc_step_number) = arma::cx_mat(arma::eye<arma::mat>(size,size),arma::zeros<arma::mat>(size,size));
								
				for (int k = 0; k < end-init; ++k){
//     				std::cout<< "loop number " << k << "from: "<< end-init << "\n";
    				//std::cout<< hamiltonian->slice(init+k) << "\n";
    				data_mgr.unitaries_cache.slice(init + k) = matrix_exp_Hamiltonian(hamiltonian->slice(init+k));
					//std::cout<< "cache slice " << k << "\n";
					data_mgr.unitaries_finished_slices.slice(calc_step_number) = data_mgr.unitaries_cache.slice(init + k)*data_mgr.unitaries_finished_slices.slice(calc_step_number);
    				//std::cout<< "finish loop " << k << "\n";
				}
			}
		}

		// calculate density matrix (don't paralleize in case lindblad equation is used.)
		#pragma omp parallel if (!do_Lindblad) shared(hamiltonian, data_mgr, iteration)
		{
			#pragma omp for
			for (int calc_step_number=0; calc_step_number < data_mgr.number_of_calc_steps; calc_step_number++){
				int init = data_mgr.calc_distro(calc_step_number);
				int end = data_mgr.calc_distro(calc_step_number+1);

				if (!do_Lindblad){

					arma::cx_mat unitary_start = arma::cx_mat(arma::eye<arma::mat>(size,size),arma::zeros<arma::mat>(size,size));
					for (int i = 0; i < calc_step_number; ++i)
					 	unitary_start = data_mgr.unitaries_finished_slices.slice(i)*unitary_start;

					data_mgr.my_density_matrices_tmp.slice(init) = unitary_start*psi0*unitary_start.t();
					
					for (int j = 0; j < end-init; ++j ){
						data_mgr.my_density_matrices_tmp.slice(j + init+ 1) = data_mgr.unitaries_cache.slice(j + init)*
										data_mgr.my_density_matrices_tmp.slice(j + init)*data_mgr.unitaries_cache.slice(j + init).t();
					}
					
					// save final unitary 
					if (calc_step_number == data_mgr.number_of_calc_steps-1){
						data_mgr.unitaries.slice(iteration) = data_mgr.unitaries_finished_slices.slice(calc_step_number)*unitary_start;
					}
				
				}else{
					std::vector<lindblad_obj>::iterator l_oper;

					for (int j = 0; j < end-init; ++j ){
						data_mgr.my_density_matrices_tmp.slice(j + init+ 1) = data_mgr.unitaries_cache.slice(j + init)*
										data_mgr.my_density_matrices_tmp.slice(j + init)*data_mgr.unitaries_cache.slice(j + init).t();
						
						for (l_oper = lindblad_oper.begin(); l_oper != lindblad_oper.end(); ++l_oper){
							data_mgr.my_density_matrices_tmp.slice(j + init+ 1) += 
								delta_t* l_oper->C * data_mgr.my_density_matrices_tmp.slice(j + init) * l_oper->C_dag +
								- delta_t* 0.5 * l_oper->C_dag_C * data_mgr.my_density_matrices_tmp.slice(j + init) +
								- delta_t* 0.5 * data_mgr.my_density_matrices_tmp.slice(j + init) * l_oper->C_dag_C;
						}
					}
					// unitaries are not saved since the linblad makes stuff non-unitary. An option would be to save the super operator (TODO?)
				}
			}
		}

		data_mgr.finish_iteration();
	}
	my_density_matrices = data_mgr.my_density_matrices/iterations;
	unitaries = data_mgr.unitaries;
	//std::cout<< average_ham<< "\n";
	//std::cout<< "average exchange: " << std::sqrt(average_exchange/1000.0) << "\n";
	//std::cout<< "average zeeman: " << std::sqrt(average_zeeman/1000.0) << "\n";
}

arma::mat DM_solver_calc_engine::return_expectation_values(arma::cx_cube input_matrices){
	arma::mat expect_val(input_matrices.n_slices, my_density_matrices.n_slices);
	
	for (uint i = 0; i < input_matrices.n_slices; ++i){
		#pragma omp parallel for
		for (uint j = 0; j < my_density_matrices.n_slices; ++j){
			expect_val(i,j) = arma::trace(arma::real(my_density_matrices.slice(j)*input_matrices.slice(i)));
			//expect_val(i,j) = arma::trace(my_density_matrices.slice(j)*input_matrices.slice(i));
		}
	}
	return expect_val;
}

arma::cx_cube DM_solver_calc_engine::get_unitaries(){
	return unitaries;
}
arma::cx_mat DM_solver_calc_engine::get_last_density_matrix(){
	return my_density_matrices.slice(my_density_matrices.n_slices-1);
}
arma::cx_cube DM_solver_calc_engine::get_all_density_matrices(){
	return my_density_matrices;
}

// int main(int argc, char const *argv[])
// {
// 	/* code */
// 	return 0;
// }