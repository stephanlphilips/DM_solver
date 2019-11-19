#include "memory_mgmnt.h"

mem_mgmt::mem_mgmt(int size_matrix){
		size = size_matrix;
		unitary = arma::cx_mat(arma::eye<arma::mat>(size,size),
				arma::zeros<arma::mat>(size,size));
		unitary_dagger = arma::cx_mat(arma::eye<arma::mat>(size,size),
				arma::zeros<arma::mat>(size,size));
	}

std::unique_ptr<unitary_obj> mem_mgmt::get_cache(int n_elem){
	// function to get a unitary in for the calculation.

	auto my_unitary_ptr = std::make_unique<unitary_obj>(
		arma::cx_mat(arma::eye<arma::mat>(size,size),
			arma::zeros<arma::mat>(size,size)),
		arma::cx_cube(arma::zeros<arma::cube>(size,size,n_elem),
			arma::zeros<arma::cube>(size,size,n_elem)));

	return my_unitary_ptr; 
}

void mem_mgmt::unitary_calc_done(int id, std::unique_ptr<unitary_obj> unitary_calculated){
	#pragma omp critical
	U_completed.insert(std::make_pair(id, std::move(unitary_calculated)));
}

bool mem_mgmt::check_U_for_calc(int to_processes_id){
	if( U_completed.find(to_processes_id) == U_completed.end() )
		return false;
	#pragma omp critical
	{
	U_completed[to_processes_id]->unitary_start = unitary;
	U_completed[to_processes_id]->unitary_start_dagger = unitary_dagger;
	unitary = U_completed[to_processes_id]->unitary_local_operation*unitary;
	unitary_dagger = unitary_dagger*U_completed[to_processes_id]->unitary_local_operation_dagger;
	}
	return true;
}

std::unique_ptr<unitary_obj> mem_mgmt::get_U_for_DM_calc(int last_processed_id){


	std::unique_ptr<unitary_obj> my_unitary_ptr = std::move(U_completed[last_processed_id]);

	#pragma omp critical
	U_completed.erase(last_processed_id);


	return my_unitary_ptr;
}