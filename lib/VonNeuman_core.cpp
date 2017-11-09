#include <armadillo>
#include <iostream>
#include <cmath>
#include <complex>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <random>

#include "math_functions.h"
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

struct noise_object_VonNeumannSolver
{	
	int type;
	magnetic_noise magnetic_noise_obj;
	one_f_noise one_f_noise_obj;
};

struct maxtrix_elem_depen_VonNeumannSolver
{
	arma::cx_mat input_matrix;
	arma::cx_mat matrix_param;
	int i;
	int j;
};

class VonNeumannSolver
{
	int size;
	int iterations = 1;
	arma::cx_cube my_density_matrices;
	std::vector<data_object_VonNeumannSolver> my_init_data;
	std::vector<maxtrix_elem_depen_VonNeumannSolver> my_parameter_depence;
	std::vector<noise_object_VonNeumannSolver> my_noise_data;
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
	void add_H1_AWG(arma::cx_mat input_matrix, double amp, double skew, double start, double stop){
		// pertubation hamiltonian where the AWG signal acts on
		data_object_VonNeumannSolver temp;
		temp.type = 2;
		temp.input_matrix1 = input_matrix;
		AWG_pulse mypulse;
		mypulse.init(amp,skew,start,stop);
		temp.AWG_obj = mypulse;
		my_init_data.push_back(temp);
	}
	void add_H1_MW_RF(arma::cx_mat input_matrix1, double rabi, double phase, double frequency, double start, double stop){
		//  input of 2 matrices, where the second one will be multiplied by the conject conjugate of the function. Note that this is an hamiltoninan supposed for the rotating frame (RF).
		data_object_VonNeumannSolver temp;
		temp.type = 3;
		temp.input_matrix1 = input_matrix1;
		phase_microwave_RWA myMWpulse;
		myMWpulse.init(rabi,phase,frequency,start,stop);
		temp.MW_obj = myMWpulse;
		my_init_data.push_back(temp);
	}
	void add_H1_element_dep_f(arma::cx_mat input_matrix, int i, int j, arma::cx_mat matrix_param){
		maxtrix_elem_depen_VonNeumannSolver temp;
		temp.input_matrix =input_matrix;
		temp.i = i;
		temp.j = j;
		temp.matrix_param = matrix_param;
		my_parameter_depence.push_back(temp);
	}
	void add_magnetic_noise(arma::cx_mat input_matrix1, double T2){
		// function to add noise, sampled from a gaussian
		noise_object_VonNeumannSolver tmp;
		tmp.type = 0;
		tmp.magnetic_noise_obj.init(input_matrix1, T2);
		my_noise_data.push_back(tmp);
	}
	void add_1f_noise(arma::cx_mat input_matrix, double noise_strength){
		noise_object_VonNeumannSolver tmp;
		tmp.type =1;
		tmp.one_f_noise_obj.init(input_matrix, noise_strength);
		my_noise_data.push_back(tmp);
	}
	void add_parameter_dependent_magnetic_noise(arma::cx_mat input_matrix, double T2, arma::cx_cube param_dep_matrix, arma::mat function_dep_param){
		noise_object_VonNeumannSolver tmp;
		tmp.type = 0;
		tmp.magnetic_noise_obj.init(input_matrix, T2, arma::conv_to<arma::Mat<int>>::from(function_dep_param), param_dep_matrix);
		my_noise_data.push_back(tmp);
	}
	void set_number_of_evalutions(int iter){
		iterations = iter;
	}

	void calculate_evolution(arma::cx_mat psi0, double start_time,double stop_time, int steps){
		my_density_matrices = arma::cx_cube(arma::zeros<arma::cube>(size,size,steps+1),arma::zeros<arma::cube>(size,size,steps+1));
		arma::cx_cube operators = arma::cx_cube(arma::zeros<arma::cube>(size,size,steps),arma::zeros<arma::cube>(size,size,steps));
		
		double delta_t  = (stop_time-start_time)/steps;
		
		// Calculate evolution matrixes without any noise:
		for (int i = 0; i < my_init_data.size(); ++i){
			switch(my_init_data[i].type){
				case 0:{
					operators.each_slice() += my_init_data[i].input_matrix1*delta_t;
					break;
				}
				case 1:{
					#pragma omp parallel for 
					for (int j = 0; j < steps; ++j){
						operators.slice(j) += my_init_data[i].input_matrix1*my_init_data[i].input_vector[j];
					}
					break;
				}
				case 2:{
					arma::cx_vec time_dep_part = my_init_data[i].AWG_obj.integrate(start_time,stop_time,steps);
					#pragma omp parallel for
					for (int j = 0; j < steps; ++j){
						operators.slice(j) += my_init_data[i].input_matrix1*time_dep_part[j];
					}
					break;
				}
				case 3:{
					arma::cx_vec time_dep_part = my_init_data[i].MW_obj.integrate(start_time,stop_time,steps);
					arma::cx_vec time_dep_part_conj = arma::conj(time_dep_part);
					arma::cx_mat input_matrix2 = my_init_data[i].input_matrix1.t();

					#pragma omp parallel for
					for (int j = 0; j < steps; ++j){
						operators.slice(j) += my_init_data[i].input_matrix1*time_dep_part[j] + input_matrix2*time_dep_part_conj[j];
					}
				}

			}
		}

		for (int i = 0; i < my_parameter_depence.size(); ++i){
			generate_parameter_dependent_matrices(&operators, my_parameter_depence[i].input_matrix, my_parameter_depence[i].i, my_parameter_depence[i].j, my_parameter_depence[i].matrix_param, delta_t);
		}
		arma::cx_cube operators_tmp = arma::cx_cube(arma::zeros<arma::cube>(size,size,steps),arma::zeros<arma::cube>(size,size,steps));
		arma::cx_cube operators_H_tmp = arma::cx_cube(arma::zeros<arma::cube>(size,size,steps),arma::zeros<arma::cube>(size,size,steps));

		for (int i = 0; i < iterations; ++i){
			operators_tmp = operators;

			for (int j = 0; j < my_noise_data.size(); ++j)
			{
				switch( my_noise_data[j].type){
					case 0:{
						operators_tmp += my_noise_data[j].magnetic_noise_obj.get_noise(&operators, steps, delta_t)*delta_t;
						break;
					}
					case 1:{
						// times delta t so it does not depend on time anymore!
						operators_tmp += my_noise_data[j].one_f_noise_obj.get_noise(&operators, steps, delta_t)*delta_t;
						break;
					}
				}
			}
			// calc matrix exponetials::
			const std::complex<double> comp(0, 1);

			#pragma omp parallel for
			for (int i = 0; i < steps; ++i){
				operators_tmp.slice(i) = custom_matrix_exp(comp*operators_tmp.slice(i));
			}

			// calc hermitian matrix
			#pragma omp parallel for
			for (int i = 0; i 
				< steps; ++i){
				operators_H_tmp.slice(i) = operators_tmp.slice(i).t();
			}
			
			arma::cx_cube my_density_matrices_tmp = arma::cx_cube(arma::zeros<arma::cube>(size,size,steps+1),arma::zeros<arma::cube>(size,size,steps+1));
			my_density_matrices_tmp.slice(0) = psi0;
			for (int i = 0; i < steps; ++i){
				my_density_matrices_tmp.slice(i+1) = operators_H_tmp.slice(i)*my_density_matrices_tmp.slice(i)*operators_tmp.slice(i);
			}
			my_density_matrices += my_density_matrices_tmp;
		}

		my_density_matrices = my_density_matrices/iterations;
		
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
};

int main(int argc, char const *argv[])
{
	/* code */
	return 0;
}