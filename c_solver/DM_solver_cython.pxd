from cyarma_lib.cyarma cimport Mat, cx_mat, cx_cube, cx_vec

cdef extern from "DM_solver_core.h":
	cdef cppclass DM_solver_calc_engine:
		DM_solver_calc_engine(double)
		void add_H1(cx_mat,cx_vec, int)
		void calculate_evolution(cx_mat, double, int)
		Mat[double] return_expectation_values(cx_cube)
		cx_mat get_unitary()
		cx_mat get_lastest_rho()
		cx_cube get_all_density_matrices()