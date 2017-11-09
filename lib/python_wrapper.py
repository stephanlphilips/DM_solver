import cyarma


cdef extern from "VonNeuman_core.h":
	cdef cppclass VonNeumannSolver:
		VonNeumannSolver(double)
		void add_H0(mat)
		void add_H1_list(mat,vec)
		void add_H1_AWG(mat, double, double,double,double)
		void add_H1_MW_RF(mat,double, double, double, double, double)
		void calculate_evolution(double,double, int)
		mat return_expectation_values(cube)

cdef class VonNeumann():
	cdef VonNeumannSolver Neum_obj
	def __cinit__(self, double size):
		self.Neum_obj = VonNeumannSolver(size)

	def add_H1_list(self, np.ndarray[ np.complex_t, ndim=2 ] input_matrix, np.ndarray[ np.complex_t, ndim=1 ] input_list):
		self.Neum_obj.add_H1_list(numpy_to_mat(input_matrix),numpy_to_vec(input_list))