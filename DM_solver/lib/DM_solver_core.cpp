#include <iostream>

#include "DM_solver_core.h"
#include "memory_mgmnt.h" 
#include "math_functions.h"

void calculate_evolution(Python_c_Interface* PythoncI){
	// per multithread steps, process 1000 unitaries
	int batch_size = 1000;

	data_manager data_mgr = data_manager(PythoncI->psi_0,PythoncI->n_elem, PythoncI->size, batch_size);

	// calculate unitaries
	#pragma omp parallel shared(PythoncI, data_mgr)
	{
		#pragma omp for
		for (int i=0; i < data_mgr.n_steps; i++){
			int init = data_mgr.calc_distro(i);
			int end  = data_mgr.calc_distro(i+1);                
            
			data_mgr.U_SliceCompleted.slice(i) = arma::cx_mat(arma::eye<arma::mat>(PythoncI->size,PythoncI->size),
															  arma::zeros<arma::mat>(PythoncI->size,PythoncI->size));
							
			for (int k = 0; k < end-init; ++k){
				data_mgr.U.slice(init + k) = matrix_exp_Hamiltonian(PythoncI->Hamiltonian.slice(init+k)*PythoncI->delta_t);
				data_mgr.U_SliceCompleted.slice(i) = 
					data_mgr.U.slice(init + k)*
					data_mgr.U_SliceCompleted.slice(i);
			}
		}
	}

	// calculate density matrix (don't paralleize in case lindblad equation is used.)
	#pragma omp parallel if (PythoncI->lindblad_opers.empty()) shared(PythoncI, data_mgr)
	{
		#pragma omp for
		for (int i=0; i < data_mgr.n_steps; i++){
			int init = data_mgr.calc_distro(i);
			int end = data_mgr.calc_distro(i+1);

			if (PythoncI->lindblad_opers.empty()){
				arma::cx_mat U_start = arma::cx_mat(arma::eye<arma::mat>(PythoncI->size,PythoncI->size),
													arma::zeros<arma::mat>(PythoncI->size,PythoncI->size));
				for (int j = 0; j < i; ++j)
				 	U_start = data_mgr.U_SliceCompleted.slice(j)*U_start;

				data_mgr.DM.slice(init) = U_start*PythoncI->psi_0*U_start.t();
				
				for (int k = 0; k < end-init; ++k ){
					data_mgr.DM.slice(k + init+ 1) = data_mgr.U.slice(k + init)*
									data_mgr.DM.slice(k + init)*data_mgr.U.slice(k + init).t();
				}
				
				// save final unitary 
				if (i == data_mgr.n_steps-1){
					data_mgr.U_final = data_mgr.U_SliceCompleted.slice(i)*U_start;
				}
			
			}else{
				std::vector<lindblad_obj>::iterator l_oper;
				for (int j = 0; j < end-init; ++j ){
					data_mgr.DM.slice(j + init+ 1) = data_mgr.U.slice(j + init)*
									data_mgr.DM.slice(j + init)*data_mgr.U.slice(j + init).t();
					
					for (l_oper = PythoncI->lindblad_opers.begin(); l_oper != PythoncI->lindblad_opers.end(); ++l_oper){
						data_mgr.DM.slice(j + init+ 1) += 
							PythoncI->delta_t* l_oper->C * data_mgr.DM.slice(j + init) * l_oper->C_dag +
							- PythoncI->delta_t* 0.5 * l_oper->C_dag_C * data_mgr.DM.slice(j + init) +
							- PythoncI->delta_t* 0.5 * data_mgr.DM.slice(j + init) * l_oper->C_dag_C;
					}
				}
				// Unitaries are not saved since the linblad makes stuff non-unitary. An option would be to save the super operator (TODO?)
			}
		}
	}
	PythoncI->iteration += 1;
	if (PythoncI->iteration == 1){
			PythoncI->DM_final = data_mgr.DM;
	}else{
		PythoncI->DM_final *= (((double) PythoncI->iteration -1) /(double) PythoncI->iteration);
		PythoncI->DM_final += data_mgr.DM / (double)PythoncI->iteration;
	}
	PythoncI->U_final = data_mgr.U_final;

	// re-init for the next calculation
	PythoncI->Hamiltonian *= 0;
	PythoncI->lindblad_opers.clear();

}