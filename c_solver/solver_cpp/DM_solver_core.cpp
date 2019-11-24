#include "DM_solver_core.h"


DM_solver_calc_engine::DM_solver_calc_engine(int size_matrix){
	size = size_matrix;
	iterations = 1;
}
void DM_solver_calc_engine::add_H1(arma::cx_mat input_matrix, arma::cx_vec time_dep_data, int hamiltonian_type, noise_specifier noise_specs){
	data_object temp;
	temp.hamiltonian_type = hamiltonian_type;
	temp.input_matrix = input_matrix;
	temp.input_vector = time_dep_data;
	temp.noise_specs = noise_specs;
	input_data.push_back(temp);
}
void DM_solver_calc_engine::set_number_of_evalutions(int iter){
	iterations = iter;
}

void DM_solver_calc_engine::calculate_evolution(arma::cx_mat psi0, double end_time, int steps){
	std::cout << "launching calc function\n";
	my_density_matrices = arma::cx_cube(arma::zeros<arma::cube>(size,size,steps+1),arma::zeros<arma::cube>(size,size,steps+1));
	unitary = arma::cx_mat(arma::zeros<arma::mat>(size,size), arma::zeros<arma::mat>(size,size));

	double delta_t  = end_time/steps;
	std::cout << "matices initialized\n";
	// Contruct basic components of hamiltonian.
	hamiltonian_constructor hamiltonian_mgr = hamiltonian_constructor(steps, size, delta_t, &input_data);
	std::cout << "hamiltonian contructed, now starting threads\n";

	for (uint i = 0; i < iterations; ++i){
		// load full hamiltonian with noise in ram.



		// start of matrix calculations. This is done using openmp jobs.

		// for the parallisation we have irregular patters, e.g. depending on the time you want to do some matrix exp/ DM 
		// multiplication of the unitary matrices, all while minimising use of memory.
		double batch_size = 1000;
		int number_of_calc_steps = std::ceil(steps/batch_size);

		// array that contains position which elements should be calculated by which thread 
		arma::Col<int> calc_distro = arma::linspace<arma::Col<int> >(0, steps, number_of_calc_steps+1);
		bool done = false;

		// Init matrices::
		arma::cx_mat unitary_tmp = arma::cx_mat(arma::eye<arma::mat>(size,size), arma::zeros<arma::mat>(size,size));
		arma::cx_cube my_density_matrices_tmp;
		arma::cx_cube* my_density_matrices_tmp_ptr;
		
		if (iterations == 1){
			my_density_matrices_tmp_ptr = &my_density_matrices;
		}else{
			my_density_matrices_tmp = arma::cx_cube(arma::zeros<arma::cube>(size,size,steps+1),arma::zeros<arma::cube>(size,size,steps+1));
			my_density_matrices_tmp_ptr = &my_density_matrices_tmp;
		}

		std::cout << "1";
		my_density_matrices_tmp_ptr->slice(0) = psi0;
		std::cout << "2";
		int num_thread = 0;

		// Counters for indicating what is already done.
		int unitaries_processed = 0;
		int elements_processed = 0;

		// initialize object that will manage the memory.
		mem_mgmt mem = mem_mgmt(size);

		std::cout << "loading hamiltonian ..\n";
		arma::cx_cube* hamiltonian = hamiltonian_mgr.load_full_hamiltonian();
		std::cout << "loading hamiltonian loaded.\n";

		std::cout << delta_t << "Number of unitaries to calculate" << number_of_calc_steps << "\n";
		#pragma omp parallel shared(hamiltonian, my_density_matrices_tmp_ptr, unitary_tmp, size)
		{
			#pragma omp single
			{

				/* Principle:
					0a) wait here until threads are available.
					0b) check if calculation is done.
					1)  check is matrix is available for DM calculation.
						 -> if so generate thread, start back at 0 else proceed to 2
					2)  check if Unitary exp. is aleady done, if not return to 0)
					3)  Make thread for matrix exp, go to 0)
				*/ 
				while(done!=true){
					// 0a)
					// TODO! Important for bigger simualtions to spare memory, no just launching all threads at the same time! Use event based trigger.
					// 0b)
					if (elements_processed == number_of_calc_steps){
						#pragma omp taskwait
						done = true;
						continue;
					}

					// 1)
					if (mem.check_U_for_calc(elements_processed)){


						// std::cout << "Starting DM_calc" << elements_processed << "\n";
						
						#pragma omp task firstprivate(elements_processed)
						{
							std::unique_ptr<unitary_obj> DM_ptr;

							DM_ptr = mem.get_U_for_DM_calc(elements_processed);

							if (elements_processed == number_of_calc_steps-1)
								unitary_tmp = (DM_ptr->unitary_start)*(DM_ptr->unitary_local_operation);
							
							
							int init = calc_distro(elements_processed);
							int n_elem = DM_ptr->hamiltonian.n_slices;

							// arma::cx_mat unitary_tmp2 = DM_ptr->unitary_start;

							my_density_matrices_tmp_ptr->slice(init) = DM_ptr->unitary_start*psi0*DM_ptr->unitary_start_dagger;
							
							// std::cout << "U\n" << unitary_tmp << "\n" << (DM_ptr->unitary_start);
							for (int j = 0; j < n_elem; ++j ){
								// unitary_tmp *= DM_ptr->hamiltonian.slice(j);
								// unitary_tmp2 *= DM_ptr->hamiltonian.slice(j);
								// my_density_matrices_tmp.slice(j + init+ 1) = unitary_tmp2*
								// 				psi0*unitary_tmp2.t(); 
								my_density_matrices_tmp_ptr->slice(j + init+ 1) = DM_ptr->hamiltonian.slice(j)*
												my_density_matrices_tmp_ptr->slice(j + init)*DM_ptr->hamiltonian.slice(j).t();
							}
							
							// std::cout << "clearing chache" << elements_processed << "\n";
						}
						++elements_processed;

						continue;
					}

					// 2)
					if (unitaries_processed == number_of_calc_steps){
						continue;
					}

					// 3)
					int init = calc_distro(unitaries_processed);
					int end = calc_distro(unitaries_processed+1);


					#pragma omp task firstprivate(unitaries_processed, init, end) //, my_init_data, my_parameter_depence, my_noise_data, delta_t)
					{
						// std::cout << "Starting unitary clac" << unitaries_processed << "\n";

						int n_elem = end-init;

						std::unique_ptr<unitary_obj> unitary_ptr =  mem.get_cache(n_elem);
						// std::cout << init << "\t" << end << "\t" << hamiltonian->n_slices << "\n";
						const std::complex<double> comp(0, 1);

						// pointer of variable of class of unique pointer?
						
						for (int k = 0; k < n_elem; ++k){
							unitary_ptr->hamiltonian.slice(k) = custom_matrix_exp(-comp*hamiltonian->slice(init+k)); //arma::expmat(-comp*hamiltonian->slice(init+k))
							unitary_ptr->unitary_local_operation = unitary_ptr->hamiltonian.slice(k)*unitary_ptr->unitary_local_operation;
							unitary_ptr->unitary_local_operation_dagger = unitary_ptr->unitary_local_operation_dagger*unitary_ptr->hamiltonian.slice(k).t();
						}

						// std::cout << unitary_ptr->unitary_local_operation << unitary_ptr->unitary_local_operation_dagger << std::endl;
						mem.unitary_calc_done(unitaries_processed, std::move(unitary_ptr));
						// std::cout << "finshed task for unitary calcualtion" << unitaries_processed << "\n";
					}

					++unitaries_processed;
				}
			}
		}


		// make if statements to only do this calculations when neccary
		if (iterations > 1)
			my_density_matrices += my_density_matrices_tmp;
		unitary += unitary_tmp;
		


		
	}
	if (iterations>1){
		my_density_matrices /= iterations;
		unitary /= iterations;
	}		
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

arma::cx_mat DM_solver_calc_engine::get_unitary(){
	return unitary;
}
arma::cx_mat DM_solver_calc_engine::get_lastest_rho(){
	return my_density_matrices.slice(my_density_matrices.n_slices-1);
}
arma::cx_cube DM_solver_calc_engine::get_all_density_matrices(){
	return my_density_matrices;
}
