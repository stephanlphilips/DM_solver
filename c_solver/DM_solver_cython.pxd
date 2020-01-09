from cyarma_lib.cyarma cimport Col, Mat, cx_mat, cx_cube, cx_vec

cdef extern from "noise_functions.h":
	struct noise_specifier:
		int noise_type
		Col[double] STD_omega;
		double STD_static

cdef extern from "DM_solver_core.h":
	cdef cppclass DM_solver_calc_engine:
		DM_solver_calc_engine(int)
		void add_H1(cx_mat,cx_vec, int, noise_specifier)
		void add_lindbladian(cx_mat A, double gamma)
		void set_number_of_evalutions(int)
		void calculate_evolution(cx_mat, double, int)
		Mat[double] return_expectation_values(cx_cube)
		cx_cube get_unitaries()
		cx_mat get_lastest_rho()
		cx_cube get_all_density_matrices()
