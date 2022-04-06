#ifndef PYTHON_C_data_interface
#define PYTHON_C_data_interface

#include<armadillo>

struct lindblad_obj{
	arma::cx_mat C; // C is \gamma*A
	arma::cx_mat C_dag;
	arma::cx_mat C_dag_C; // premultiplied matrices
	arma::cx_mat C_C_dag; // premultiplied matrices
};

class Python_c_Interface{
    public:
		int size;
		int iteration;
		double n_elem;
		double delta_t;

		arma::cx_mat psi_0;
		arma::cx_cube Hamiltonian;
		std::vector<lindblad_obj> lindblad_opers;		

		arma::cx_cube DM_final;
		arma::cx_mat  U_final;
		arma::vec expectation_cache;

		Python_c_Interface(std::complex<double>* DM_data, int n_rows, int n_cols, int n_step, double t_step);
		void add_H(std::complex<double>* data2D, int n_rows, int n_cols, 
		            	std::complex<double>* data1D, int n_elements);
		void add_lindbladian(std::complex<double>* data2D, int n_rows, int n_cols, double gamma);
		arma::vec return_expectation_value(std::complex<double>* mOperatorData, int n_rows, int n_cols);
		arma::cx_mat return_unitary();
};

extern "C"{
Python_c_Interface* DM_new(std::complex<double>* DM_data,
					 int n_rows, int n_cols, int n_step, double t_step);
void DM_add_H(Python_c_Interface* DM, std::complex<double>* data2D, int n_rows, int n_cols, 
                                std::complex<double>* data1D, int n_elements);
void DM_add_lindbladian(Python_c_Interface* DM, std::complex<double>* data2D, int n_rows,
							int n_cols, double gamma);
double* DM_return_expectation_value(Python_c_Interface* DM, std::complex<double>* mOperatorData,
							int n_rows, int n_cols);
std::complex<double>* DM_return_unitary(Python_c_Interface* DM);
int  DM_n_elem(Python_c_Interface* DM);
void DM_del(Python_c_Interface* DM);
}

#endif