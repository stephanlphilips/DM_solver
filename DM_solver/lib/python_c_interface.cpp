#include<cmath>
#include<complex> 
#include<vector>
#include <stdexcept>

#include "python_c_interface.h"
#include "DM_solver_core.h"

Python_c_Interface::Python_c_Interface(std::complex<double>* DM_data, int n_rows, int n_cols, int n_step, double t_step){
	size  = n_rows;
	n_elem = n_step;
	delta_t = t_step;
	iteration = 0;

	psi_0 = arma::cx_mat(DM_data, n_rows, n_cols, true,  false);
	Hamiltonian = arma::cx_cube(arma::zeros<arma::cube>(size,size,n_elem),arma::zeros<arma::cube>(size,size,n_elem));
};

void Python_c_Interface::add_H(std::complex<double>* data2D, int n_rows, int n_cols, 
                        	std::complex<double>* data1D, int n_elements){
	if (n_elem != n_elements){
		throw std::runtime_error(std::string("Predefined size and nuber of elements are different!"));
	}

	arma::cx_mat Oper = arma::cx_mat(data2D, n_rows, n_cols, true,  false);
	arma::cx_vec amps = arma::cx_vec(data1D, n_elements, true,  false);

	int end_loop = n_elem;
	
	#pragma omp parallel for
	for (int i = 0; i < end_loop; ++i){
		// this done in case complex arguments in the amps are given.
		Hamiltonian.slice(i) += 
			trimatu(Oper) * amps.at(i) + trimatl(Oper,-1) * std::conj(amps.at(i));
	}
};

void Python_c_Interface::add_lindbladian(std::complex<double>* data2D, int n_rows, int n_cols, double gamma){
	arma::cx_mat A = arma::cx_mat(data2D, n_rows, n_cols, true,  false);

	lindblad_obj lind_tmp;
	lind_tmp.C = gamma*A;
	lind_tmp.C_dag = lind_tmp.C.t();
	lind_tmp.C_dag_C = lind_tmp.C_dag * lind_tmp.C;

	lindblad_opers.push_back(lind_tmp);
};

arma::vec Python_c_Interface::return_expectation_value(std::complex<double>* mOperatorData, int n_rows, int n_cols){
	arma::cx_mat m_operator = arma::cx_mat(mOperatorData, n_rows, n_cols, true,  false);

	arma::vec expect_val = arma::vec(DM_final.n_slices);
	
	#pragma omp parallel for
	for (uint j = 0; j < DM_final.n_slices; ++j){
		expect_val.at(j) = arma::trace(arma::real(DM_final.slice(j)*m_operator));
	}

	return expect_val;
}

arma::cx_mat Python_c_Interface::return_unitary(){
	return U_final;
}


Python_c_Interface* DM_new(std::complex<double>* DM_data,
					 int n_rows, int n_cols, int n_step, double t_step){
	return new Python_c_Interface(DM_data, n_rows, n_cols, n_step, t_step);
};

void DM_add_H(Python_c_Interface* DM, std::complex<double>* data2D, int n_rows, int n_cols, 
                                std::complex<double>* data1D, int n_elements){
    DM->add_H(data2D, n_rows, n_cols, data1D, n_elements);
};

void DM_add_lindbladian(Python_c_Interface* DM, std::complex<double>* data2D, int n_rows,
							int n_cols, double gamma){
    DM->add_lindbladian(data2D, n_rows, n_cols, gamma);
};

double* DM_return_expectation_value(Python_c_Interface* DM, std::complex<double>* mOperatorData,
							int n_rows, int n_cols){
	DM->expectation_cache = DM->return_expectation_value(mOperatorData, n_rows, n_cols);
	return DM->expectation_cache.memptr();
};

std::complex<double>* DM_return_unitary(Python_c_Interface* DM){
	return DM->U_final.memptr();
};


int DM_n_elem(Python_c_Interface* DM){
	return DM->n_elem;
};

void DM_del(Python_c_Interface* DM){
	delete DM;
};
