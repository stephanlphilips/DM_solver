import pathlib
import DM_solver.lib
import numpy as np

from ctypes import cdll, c_void_p, c_int, c_double, memmove
lib = cdll.LoadLibrary(str(pathlib.Path(DM_solver.lib.__file__).parent.resolve()) + "/c_libraries_DM_solver.so")  

lib.DM_new.argtypes = [c_void_p, c_int, c_int, c_int, c_double]
lib.DM_new.restype = c_void_p

lib.DM_add_H.argtypes = [c_void_p, c_void_p, c_int, c_int, c_void_p, c_int]
lib.DM_add_H.restype = None

lib.DM_add_lindbladian.argtypes = [c_void_p, c_void_p, c_int, c_int, c_double]
lib.DM_add_lindbladian.restype = None

lib.DM_return_expectation_value.argtypes = [c_void_p, c_void_p, c_int, c_int]
lib.DM_return_expectation_value.restype = c_void_p

lib.DM_return_unitary.argtypes = [c_void_p]
lib.DM_return_unitary.restype = c_void_p

lib.DM_n_elem.argtypes = [c_void_p]
lib.DM_n_elem.restype = c_int

lib.DM_del.argtypes = [c_void_p]
lib.DM_del.restype = None

lib.calculate_evolution.argtypes = [c_void_p]
lib.calculate_evolution.restype = None


def to_c_array(array):
	# keep array variable in memory so garbage collection does not remove it before usage.
	array = np.asarray(np.asfortranarray(array), dtype=np.complex128)
	return [array.ctypes.data, *array.shape], array

class python_c_interface():
	def __init__(self, psi_0, t_end, sample_rate):
		self.t_end = t_end
		self.sample_rate = sample_rate
		self.psi_0 = psi_0
		psi_0_raw, _  = to_c_array(psi_0)
		self.raw_data = lib.DM_new(*psi_0_raw ,round(t_end*sample_rate), 1/sample_rate)

	def add_H_channel_data(self, H_channel):
		operator, _ =  to_c_array(H_channel.matrix)
		pulseData, __ =  to_c_array(H_channel.render_pulse(self.t_end, self.sample_rate))
		lib.DM_add_H(self.raw_data, *operator, *pulseData)

	def add_lindbladian(self, matrix, gamma):
		lib.DM_add_lindbladian(self.raw_data, *to_c_array(matrix), gamma)

	def calculate_evolution(self):
		lib.calculate_evolution(self.raw_data)

	def get_expectation_values(self, mOperator):
		mOperator, _ =  to_c_array(mOperator)
		c_ptr = lib.DM_return_expectation_value(self.raw_data, *mOperator)
		res = bytearray(8* lib.DM_n_elem(self.raw_data))
		r_ptr = (c_double * lib.DM_n_elem(self.raw_data)).from_buffer(res)
		memmove(r_ptr, c_ptr, 8*lib.DM_n_elem(self.raw_data))
		return np.frombuffer(buffer=res)

	def get_unitary(self):
		c_ptr = lib.DM_return_unitary(self.raw_data)
		res = bytearray(16* lib.DM_n_elem(self.raw_data))
		r_ptr = (c_double * lib.DM_n_elem(self.raw_data)).from_buffer(res)
		memmove(r_ptr, c_ptr, 16*lib.DM_n_elem(self.raw_data))
		return np.ndarray(self.psi_0.shape, buffer=res, dtype=np.complex128, order='F')

	def __del__(self):
		lib.DM_del(self.raw_data)

if __name__ == '__main__':
	from DM_solver.solver import H_channel

	psi_0 = np.matrix([[1,0],[0,0]])
	Z  = np.matrix([[1,0.],
					[0.,1]])/2

	t_end = 10e-9
	sample_rate = 1e9

	c_data = python_c_interface(psi_0, t_end, sample_rate)
	c_data.add_H_channel_data(H_channel(Z, 1e9))
	# c_data.add_lindbladian(Z, 10)
	c_data.calculate_evolution()
	print(c_data.get_expectation_values(Z))
	print(c_data.get_unitary())
