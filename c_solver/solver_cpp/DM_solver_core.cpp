#include "DM_solver_core.h"


DM_solver_calc_engine::DM_solver_calc_engine(int size_matrix){
	size = size_matrix;
	iterations = 1;
	do_Lindblad = false;
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
void DM_solver_calc_engine::set_number_of_evalutions(int iter){
	iterations = iter;
}

void DM_solver_calc_engine::calculate_evolution(arma::cx_mat psi0, double end_time, int steps){
	// per multithread steps, process 1000 unitaries
	int batch_size = 1000;
	int delta_t = end_time/steps;

	hamiltonian_constructor hamiltonian_mgr = hamiltonian_constructor(steps, size, delta_t, &input_data);
	data_manager data_mgr = data_manager(steps,size, iterations, batch_size);

	for (int iteration = 0; iteration < iterations; ++iteration){
		std::cout<< "iterations " << iteration << "\n";
		
		data_mgr.init_iteration(psi0);
		arma::cx_cube* hamiltonian = hamiltonian_mgr.load_full_hamiltonian();

		// calculate unitaries
		#pragma omp parallel shared(hamiltonian, data_mgr)
		{
			#pragma omp for
			for (int calc_step_number=0; calc_step_number < data_mgr.number_of_calc_steps; calc_step_number++){
				int init = data_mgr.calc_distro(calc_step_number);
				int end = data_mgr.calc_distro(calc_step_number+1);

				data_mgr.unitaries_finished_slices.slice(calc_step_number) = arma::cx_mat(arma::eye<arma::mat>(size,size),arma::zeros<arma::mat>(size,size));
				
				for (int k = 0; k < end-init; ++k){
					data_mgr.unitaries_cache.slice(init + k) = matrix_exp_Hamiltonian(hamiltonian->slice(init+k));
					// data_mgr.unitaries_cache.slice(init + k) = custom_matrix_exp(-comp*hamiltonian->slice(init+k));
					data_mgr.unitaries_finished_slices.slice(calc_step_number) = data_mgr.unitaries_cache.slice(init + k)*data_mgr.unitaries_finished_slices.slice(calc_step_number);
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
						data_mgr.unitaries.slice(iteration) = unitary_start*data_mgr.unitaries_finished_slices.slice(calc_step_number);
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
				}
			}
		}

		data_mgr.finish_iteration();
	}
	my_density_matrices = data_mgr.my_density_matrices/iterations;
	unitaries = data_mgr.unitaries;
}

arma::mat DM_solver_calc_engine::return_expectation_values(arma::cx_cube input_matrices){
	arma::mat expect_val(input_matrices.n_slices, my_density_matrices.n_slices);
	
	for (uint i = 0; i < input_matrices.n_slices; ++i){
		#pragma omp parallel for
		for (uint j = 0; j < my_density_matrices.n_slices; ++j){
			expect_val(i,j) = arma::trace(arma::real(my_density_matrices.slice(j)*input_matrices.slice(i)));
		}
	}
	return expect_val;
}

arma::cx_mat DM_solver_calc_engine::get_unitaries(){
	return unitaries.slice(0);
}
arma::cx_mat DM_solver_calc_engine::get_lastest_rho(){
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