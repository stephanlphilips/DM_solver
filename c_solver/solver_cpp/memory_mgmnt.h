#ifndef MEMORY_MANAGEMENT_H
#define MEMORY_MANAGEMENT_H

#include <armadillo>
#include <memory>
#include <map>

/*
Struct holding all the matrices needed for the density matrix calculations.
This is all temporary data:
	* unitary describing the evolution of all the chunks before this one.
	* unitary describing the evolution of the calculated chunk.
	* Hamiltonian of each time step. Is later on exponentiated to convert to a unitary.
*/
struct unitary_obj
{
	arma::cx_mat unitary_start;
	arma::cx_mat unitary_start_dagger;
	arma::cx_mat unitary_local_operation;
	arma::cx_mat unitary_local_operation_dagger;
	arma::cx_cube hamiltonian;

	unitary_obj(arma::cx_mat unitary_tmp, arma::cx_cube hamiltonian_init){
		unitary_local_operation = unitary_tmp;
		unitary_local_operation_dagger = unitary_tmp;
		unitary_start = unitary_tmp;
		unitary_start_dagger = unitary_tmp;
		hamiltonian = hamiltonian_init;
	};
};

class mem_mgmt
{
	std::map<int, std::unique_ptr<unitary_obj> > U_completed;
	arma::cx_mat unitary;
	arma::cx_mat unitary_dagger;
	int size;
public:
	mem_mgmt(int size_matrix);
	std::unique_ptr<unitary_obj> get_cache(int n_elem);

	/* function that check if there is a unitary available for further calculation.
	Note that this function also evolves the global unitary in the memory class to make sure 
	that paralelisation of the unitaries is possible. 
	*/
	void unitary_calc_done(int id, std::unique_ptr<unitary_obj> unitary_calculated);
	bool check_U_for_calc(int to_processes_id);
	std::unique_ptr<unitary_obj> get_U_for_DM_calc(int last_processed_id);
};

#endif