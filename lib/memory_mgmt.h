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

/* 
Object for holding all the construction data for the hamiltonian.
*/
struct data_object_VonNeumannSolver
{
	int type;
	arma::cx_mat input_matrix1;
	arma::cx_vec input_vector;
	phase_microwave_RWA MW_obj;
	AWG_pulse AWG_obj;
	
};

/*
Object that hold matrix dependencies. See manual for more detail on this functionality.
*/
struct maxtrix_elem_depen_VonNeumannSolver
{
	int type;
	int i;
	int j;
	arma::Mat<int> locations;

	arma::cx_mat input_matrix;
	arma::cx_mat matrix_param;
	double frequency;

	arma::cx_vec amplitudes;
};


class mem_mgmt
{
	// Array that holds data of completed Unitaries
	std::map<int, std::unique_ptr<unitary_obj> > U_completed;


	arma::cx_mat unitary;
	arma::cx_mat unitary_dagger;
	int size;
	

public:
	mem_mgmt(int size_matrix){
		size = size_matrix;
		unitary = arma::cx_mat(arma::eye<arma::mat>(size,size),
				arma::zeros<arma::mat>(size,size));
		unitary_dagger = arma::cx_mat(arma::eye<arma::mat>(size,size),
				arma::zeros<arma::mat>(size,size));
	}


	
	std::unique_ptr<unitary_obj> get_cache(int n_elem){
		// function to get a unitary in for the calculation.

		auto my_unitary_ptr = std::make_unique<unitary_obj>(
			arma::cx_mat(arma::eye<arma::mat>(size,size),
				arma::zeros<arma::mat>(size,size)),
			arma::cx_cube(arma::zeros<arma::cube>(size,size,n_elem),
				arma::zeros<arma::cube>(size,size,n_elem)));

	return std::move(my_unitary_ptr); 
	}

	void unitary_calc_done(int id, std::unique_ptr<unitary_obj> unitary_calculated){
		#pragma omp critical
		U_completed.insert(std::make_pair(id, std::move(unitary_calculated)));
	}

	/* function that check if there is a unitary available for further calculation.
	Note that this function also evolves the global unitary in the memory class to make sure 
	that paralelisation of the unitaries is possible. 
	*/
	bool check_U_for_calc(int to_processes_id){
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

	std::unique_ptr<unitary_obj> get_U_for_DM_calc(int last_processed_id){


		std::unique_ptr<unitary_obj> my_unitary_ptr = std::move(U_completed[last_processed_id]);

		#pragma omp critical
		U_completed.erase(last_processed_id);


		return std::move(my_unitary_ptr);
	}
};