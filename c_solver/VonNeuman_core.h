#include <armadillo>
#include <iostream>
#include <cmath>
#include <complex>
#include <math.h>
#include "DspFilters/Dsp.h"
#include <random>
#include <chrono>
#include <string>
#include <memory>
#include <map>
#include "math_functions.h"
#include "signal_functions.h"
#include "memory_mgmt.h"
#include "dependency_functions.h"
#include "noise_functions.h"
#include "hamiltonian_constructor.h"
// #include "omp.h"
#include <chrono> 


class VonNeumannSolver
{
	int size;
	int iterations = 1;
	arma::cx_cube my_density_matrices;
	arma::cx_mat unitary;
	std::vector<data_object_VonNeumannSolver> my_init_data;
	std::vector<maxtrix_elem_depen_VonNeumannSolver> my_parameter_depence;
	std::vector<noise> my_noise_data;
public:
	VonNeumannSolver(int size_matrix){
		size= size_matrix;
	}
	void add_H0(arma::cx_mat input_matrix){
		// Constant hamiltonian part, does not change in time.
		data_object_VonNeumannSolver temp;
		temp.type = 0;
		temp.input_matrix1 = input_matrix;
		my_init_data.push_back(temp);
	}
	void add_H1_list(arma::cx_mat input_matrix, arma::cx_vec time_dep_data){
		// Amplide of hamiltonian changes in time, the list is supposed to contain the already integrated time data
		data_object_VonNeumannSolver temp;
		temp.type = 1;
		temp.input_matrix1 = input_matrix;
		temp.input_vector = time_dep_data;
		my_init_data.push_back(temp);
	}
	// void add_H1_AWG(arma::cx_mat input_matrix, double amp, double skew, double start, double stop){
	// 	// pertubation hamiltonian where the AWG signal acts on
	// 	data_object_VonNeumannSolver temp;
	// 	temp.type = 2;
	// 	temp.input_matrix1 = input_matrix;
	// 	AWG_pulse mypulse;
	// 	mypulse.init(amp,skew,start,stop);
	// 	temp.AWG_obj = mypulse;
	// 	my_init_data.push_back(temp);o
	// }

	void add_H1_AWG(arma::mat pulse_data, arma::cx_mat input_matrix){
		// Add a time dependent signal that you would send down through an awg channel to your sample.
		data_object_VonNeumannSolver temp;
		temp.type = 2;
		temp.AWG_obj.init(pulse_data,input_matrix);
		my_init_data.push_back(temp);
	}

	void add_H1_AWG(arma::mat pulse_data, arma::cx_mat input_matrix, arma::mat filter_coeff){
		data_object_VonNeumannSolver temp;
		temp.type = 2;
		temp.AWG_obj.init(pulse_data,input_matrix, filter_coeff);

		my_init_data.push_back(temp);
	}

	void add_H1_MW_RWA(arma::cx_mat input_matrix1, double rabi, double phase, double frequency, double start, double stop){
		//  input 1 of 2 matrices, where the second one will be multiplied by the conject conjugate of the function. 
		data_object_VonNeumannSolver temp;
		temp.type = 3;
		phase_microwave_RWA myMWpulse;
		myMWpulse.init(rabi,phase,frequency,start,stop, input_matrix1);
		temp.MW_obj_RWA = myMWpulse;
		my_init_data.push_back(temp);
	}
	void add_H1_MW_obj_RWA(phase_microwave_RWA my_mwobject){
		data_object_VonNeumannSolver temp;
		temp.type = 3;
		temp.MW_obj_RWA = my_mwobject;
		my_init_data.push_back(temp);
	}

	void add_H1_MW_obj(MW_pulse my_mwobject){
		data_object_VonNeumannSolver temp;
		temp.type = 4;
		temp.MW_obj = my_mwobject;
		my_init_data.push_back(temp);
	}

	void add_H1_element_dep_f(arma::cx_mat input_matrix, int i, int j, arma::cx_mat matrix_param){
		maxtrix_elem_depen_VonNeumannSolver temp;
		temp.input_matrix =input_matrix;
		temp.type = 0;
		temp.i = i;
		temp.j = j;
		temp.matrix_param = matrix_param;
		my_parameter_depence.push_back(temp);
	}
	void mk_param_time_dep(arma::Mat<int> locations, double frequency){
		// Note that you only need to specify the things in the upper right part of a matrix. The rest is automatically generated.
		// Not it is not itendeded to specify stuff on the diagonal (you should'nt anyway).
		// locations[a,b], where a'th element contains a location where you want do apply a frequency f and b (0,1) is the (i,j)th element in the matrix
		
		maxtrix_elem_depen_VonNeumannSolver temp;
		temp.type = 1;
		temp.locations =locations;
		temp.frequency = frequency;
		my_parameter_depence.push_back(temp);
	}

	void add_static_gauss_noise(arma::cx_mat input_matrix1, double T2){
		// function to add noise, sampled from a gaussian
		noise mynoise;
		mynoise.init_gauss(input_matrix1, T2);
		my_noise_data.push_back(mynoise);
	}

	void add_1f_noise(arma::cx_mat input_matrix, double noise_strength, double alpha){
		noise mynoise;
		mynoise.init_pink(input_matrix, noise_strength, alpha);
		my_noise_data.push_back(mynoise);
	}

	void add_white_noise(arma::cx_mat input_matrix, double noise_strength){
		noise mynoise;
		mynoise.init_white(input_matrix, noise_strength);
		my_noise_data.push_back(mynoise);
	}

	void add_noise_object(noise noise_obj){
		my_noise_data.push_back(noise_obj);
	}
	
	void set_number_of_evalutions(int iter){
		iterations = iter;
	}

	void calculate_evolution(arma::cx_mat psi0, double start_time,double stop_time, int steps){
		my_density_matrices = arma::cx_cube(arma::zeros<arma::cube>(size,size,steps+1),arma::zeros<arma::cube>(size,size,steps+1));
		unitary = arma::cx_mat(arma::zeros<arma::mat>(size,size), arma::zeros<arma::mat>(size,size));

		double delta_t  = (stop_time-start_time)/steps;

		// Contruct basic components of hamiltonian.
		preload_hamilonian(&my_init_data, start_time, stop_time, steps);

		for (int i = 0; i < iterations; ++i){
			// start of matrix calculations. This is done using openmp jobs.

			// for the parallisation we have irregular patters, e.g. depending on the time you want to do some matrix exp/ DM 
			// multiplication of the unitary matrices, all while minimising use of memory.
			double batch_size = 1000;
			int number_of_calc_steps = std::ceil(steps/batch_size);

			// array that contains position which elements should be calculated by which thread 
			arma::Col<int> calc_distro = arma::linspace<arma::Col<int>>(0, steps, number_of_calc_steps+1);
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

			my_density_matrices_tmp_ptr->slice(0) = psi0;
			int num_thread = 0;
			// int max_num_threads = omp_get_max_threads();

			// Counters for indicating what is already done.
			int unitaries_processed = 0;
			int elements_processed = 0;

			// initialize object that will manage the memory.
			mem_mgmt mem = mem_mgmt(size);

			// Preload noise:
			preload_noise(&my_noise_data, start_time, stop_time, steps);

			// std::cout << delta_t << "Number of unitaries to calculate" << number_of_calc_steps << "\n";
			#pragma omp parallel // shared(my_density_matrices_tmp_ptr, unitary_tmp, size)
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
								
								
								int const init = calc_distro(elements_processed);
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
							// // std::cout << init << "\t" << end << "\t" << unitary_ptr->hamiltonian.n_slices << "\n";
							const std::complex<double> comp(0, 1);

							double start_time = init*delta_t;
							double end_time = end*delta_t;

							// pointer of variable of class of unique pointer?
							contruct_hamiltonian( &(unitary_ptr->hamiltonian), init, end, 
								start_time, end_time, delta_t,
								&my_init_data, &my_parameter_depence, &my_noise_data);
							for (int k = 0; k < n_elem; ++k){
								unitary_ptr->hamiltonian.slice(k) = custom_matrix_exp(-comp*unitary_ptr->hamiltonian.slice(k));
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

	arma::mat return_expectation_values(arma::cx_cube input_matrices){
		arma::mat expect_val(input_matrices.n_slices, my_density_matrices.n_slices);
		
		for (int i = 0; i < input_matrices.n_slices; ++i){
			#pragma omp parallel for
			for (int j = 0; j < my_density_matrices.n_slices; ++j){
				expect_val(i,j) = arma::trace(arma::real(my_density_matrices.slice(j)*input_matrices.slice(i)));
			}
		}
		return expect_val;
	}

	arma::cx_mat get_unitary(){
		return unitary;
	}
	arma::cx_mat get_lastest_rho(){
		return my_density_matrices.slice(my_density_matrices.n_slices-1);
	}
	arma::cx_cube get_all_density_matrices(){
		return my_density_matrices;
	}
};
