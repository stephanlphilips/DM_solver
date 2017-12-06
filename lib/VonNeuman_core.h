#include <armadillo>
#include <iostream>
#include <cmath>
#include <complex>
#include <math.h>
#include "DspFilters/Dsp.h"
#include <random>
#include <chrono>
#include <string>
#include "math_functions.h"
#include "dependency_functions.h"
#include "signal_functions.h"
#include "noise_functions.h"


struct data_object_VonNeumannSolver
{
	int type;
	arma::cx_mat input_matrix1;
	arma::cx_vec input_vector;
	phase_microwave_RWA MW_obj;
	AWG_pulse AWG_obj;
	
};

struct maxtrix_elem_depen_VonNeumannSolver
{
	int type;
	int i;
	int j;
	arma::Mat<int> locations;

	arma::cx_mat input_matrix;
	arma::cx_mat matrix_param;
	double frequency;
};

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
	// 	my_init_data.push_back(temp);
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

	void add_H1_MW (arma::cx_mat input_matrix1, double rabi, double phase, double frequency, double start, double stop){
		//  input of 2 matrices, where the second one will be multiplied by the conject conjugate of the function. 
		data_object_VonNeumannSolver temp;
		temp.type = 3;
		temp.input_matrix1 = input_matrix1;
		phase_microwave_RWA myMWpulse;
		myMWpulse.init(rabi,phase,frequency,start,stop);
		temp.MW_obj = myMWpulse;
		my_init_data.push_back(temp);
	}
	void add_H1_MW_obj(arma::cx_mat input_matrix1, phase_microwave_RWA my_mwobject){
		data_object_VonNeumannSolver temp;
		temp.type = 3;
		temp.input_matrix1 = input_matrix1;
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
		arma::cx_cube operators = arma::cx_cube(arma::zeros<arma::cube>(size,size,steps),arma::zeros<arma::cube>(size,size,steps));
		unitary = arma::cx_mat(arma::zeros<arma::mat>(size,size), arma::zeros<arma::mat>(size,size));

		double delta_t  = (stop_time-start_time)/steps;
		
		// Calculate evolution matrixes without any noise:
		for (int i = 0; i < my_init_data.size(); ++i){
			switch(my_init_data[i].type){
				case 0:{
					operators.each_slice() += my_init_data[i].input_matrix1*delta_t;
					break;
				}
				case 1:{
					for (int j = 0; j < steps; ++j){
						operators.slice(j) += my_init_data[i].input_matrix1*my_init_data[i].input_vector[j];
					}
					break;
				}
				case 2:{
<<<<<<< HEAD
					my_init_data[i].AWG_obj.integrate(&operators, start_time,stop_time, steps);
=======
					arma::cx_vec time_dep_part = my_init_data[i].AWG_obj.integrate(start_time,stop_time,steps);
					for (int j = 0; j < steps; ++j){
						operators.slice(j) += my_init_data[i].input_matrix1*time_dep_part[j];
					}
>>>>>>> origin/new_noise_model
					break;
				}
				case 3:{
					arma::cx_vec time_dep_part = my_init_data[i].MW_obj.integrate(start_time,stop_time,steps);
					arma::cx_vec time_dep_part_conj = arma::conj(time_dep_part);
					arma::cx_mat input_matrix2 = my_init_data[i].input_matrix1.t();

					for (int j = 0; j < steps; ++j){
						operators.slice(j) += my_init_data[i].input_matrix1*time_dep_part[j] + input_matrix2*time_dep_part_conj[j];
					}
				}

			}
		}

		for (int i = 0; i < my_parameter_depence.size(); ++i){
			if (my_parameter_depence[i].type ==0)
				generate_parameter_dependent_matrices(&operators, my_parameter_depence[i].input_matrix, my_parameter_depence[i].i, my_parameter_depence[i].j, my_parameter_depence[i].matrix_param, delta_t);
		}
		for (int i = 0; i < my_parameter_depence.size(); ++i){
			if (my_parameter_depence[i].type ==1)
				generate_time_dependent_matrices(&operators, my_parameter_depence[i].locations, my_parameter_depence[i].frequency, start_time, stop_time, steps);
		}


		#pragma omp parallel for
		for (int i = 0; i < iterations; ++i){
			arma::cx_cube operators_tmp = arma::cx_cube(arma::zeros<arma::cube>(size,size,steps),arma::zeros<arma::cube>(size,size,steps));
			arma::cx_cube operators_H_tmp = arma::cx_cube(arma::zeros<arma::cube>(size,size,steps),arma::zeros<arma::cube>(size,size,steps));
			operators_tmp = operators;

			// Make sure that there are no two writes at the same time.
			for (int j = 0; j < my_noise_data.size(); ++j){
				operators_tmp += my_noise_data[j].get_noise(&operators, steps, delta_t)*delta_t;
				}
			// calc matrix exponetials::
			const std::complex<double> comp(0, 1);

			for (int i = 0; i < steps; ++i){
				operators_tmp.slice(i) = custom_matrix_exp(-comp*operators_tmp.slice(i));
			}

			// calc hermitian matrix
			for (int i = 0; i < steps; ++i){
				operators_H_tmp.slice(i) = operators_tmp.slice(i).t();
			}

			arma::cx_mat unitary_tmp = arma::cx_mat(arma::eye<arma::mat>(size,size), arma::zeros<arma::mat>(size,size));
			arma::cx_mat unitary_tmp2 = arma::cx_mat(arma::eye<arma::mat>(size,size), arma::zeros<arma::mat>(size,size));
			arma::cx_cube my_density_matrices_tmp = arma::cx_cube(arma::zeros<arma::cube>(size,size,steps+1),arma::zeros<arma::cube>(size,size,steps+1));
			my_density_matrices_tmp.slice(0) = psi0;

			for (int i = 0; i < steps; ++i){
				unitary_tmp = unitary_tmp*operators_tmp.slice(i);
				my_density_matrices_tmp.slice(i+1) = operators_tmp.slice(i)*my_density_matrices_tmp.slice(i)*operators_H_tmp.slice(i);
			}

			// Make sure that there are no two writes at the same time.
			#pragma omp critical
			{
				my_density_matrices += my_density_matrices_tmp;
				unitary += unitary_tmp;
			}
		}

		my_density_matrices /= iterations;
		unitary /= iterations;
		
	}

	arma::mat return_expectation_values(arma::cx_cube input_matrices){
		arma::mat expect_val(input_matrices.n_slices, my_density_matrices.n_slices);
		
		for (int i = 0; i < input_matrices.n_slices; ++i){
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

