import numpy as np
import scipy as sp

import sys
import  c_solver.ME_solver as ME

import matplotlib.pyplot as plt
from qutip import *
import qutip as qt
import cmath

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=180)


class double_dot_hamiltonian():
	def __init__(self, B_1, B_2, chargingE1, chargingE2, tunnel_coupling):
		# Generate static part of hamiltonian (using numpy/qutip stuff)
		# All energy's should be expressed in Hz
		# H_bar = 1
		self.H_charg1 = np.array(list(basis(6,4)*basis(6,4).dag()))[:,0]
		self.H_charg2 = np.array(list(basis(6,5)*basis(6,5).dag()))[:,0]


		self.H_B_field1 = np.array(list(-basis(6,0)*basis(6,0).dag() - basis(6,1)*basis(6,1).dag() +
							basis(6,2)*basis(6,2).dag() + basis(6,3)*basis(6,3).dag()))[:,0]/2
		self.H_B_field2 = np.array(list(-basis(6,0)*basis(6,0).dag() + basis(6,1)*basis(6,1).dag() -
							basis(6,2)*basis(6,2).dag() + basis(6,3)*basis(6,3).dag()))[:,0]/2

		self.H_exchange = np.array(
			list(-basis(6,1)*basis(6,1).dag() -basis(6,2)*basis(6,2).dag() +
				basis(6,1)*basis(6,2).dag() + basis(6,2)*basis(6,1).dag()))[:,0]/2
		self.H_tunnel = np.array(list(basis(6,1)*basis(6,4).dag() - basis(6,2)*basis(6,4).dag() +
									basis(6,1)*basis(6,5).dag() - basis(6,2)*basis(6,5).dag() +
									basis(6,4)*basis(6,1).dag() - basis(6,4)*basis(6,2).dag() +
									basis(6,5)*basis(6,1).dag() - basis(6,5)*basis(6,2).dag()))[:,0]

		self.B_1 = B_1
		self.B_2 = B_2
		self.chargingE1 = chargingE1
		self.chargingE2 = chargingE2
		self.tunnel_coupling = tunnel_coupling
		self.my_Hamiltonian = 2*np.pi*(self.H_charg1*chargingE1 + self.H_charg2*chargingE2 + self.H_tunnel*tunnel_coupling)
		self.h_raw = self.my_Hamiltonian/2/np.pi + self.H_B_field1* B_1 + self.H_B_field2*B_2
		# Clock of rotating frame
		self.f_qubit1 = B_1
		self.f_qubit2 = B_2

		# Init params
		self.amp_noise_1 = 0
		self.amp_noise_2 = 0
		self.number_of_samples = 1
		self.one_f_noise = 0

		# Create the solver
		self.solver_obj = ME.VonNeumann(6)
		# Add the init hamiltonian
		self.solver_obj.add_H0(self.my_Hamiltonian)

		if self.f_qubit1-self.f_qubit2 != 0:
			# add time dependent tunnelcouplings (see presentation)
			locations_1 = np.array([[1,4],[1,5]],dtype=np.int32)
			self.solver_obj.add_cexp_time_dep(locations_1, (self.f_qubit1-self.f_qubit2)/2)

			locations_1 = np.array([[2,4],[2,5]],dtype=np.int32)
			self.solver_obj.add_cexp_time_dep(locations_1, (self.f_qubit2-self.f_qubit1)/2)
	
	# Functions that can simulate the effect of MW signals and pulses to the sample.
	# Quasi dirty implementation of gaussian pulses.
	def mw_pulse(self, freq, phase, rabi_f, t_start, t_stop, sigma_Gauss= None, RWA = True):
		self.H_mw_qubit_1 = np.array(list(basis(6,0)*basis(6,2).dag() + basis(6,1)*basis(6,3).dag()))[:,0]
		self.H_mw_qubit_2 = np.array(list(basis(6,0)*basis(6,1).dag() + basis(6,2)*basis(6,3).dag()))[:,0]

		if RWA == True:
			if sigma_Gauss == None:
				# Note RWA
				self.solver_obj.add_H1_MW_RF_RWA(self.H_mw_qubit_1, rabi_f*(np.pi), phase, freq-self.f_qubit1, t_start, t_stop)
				self.solver_obj.add_H1_MW_RF_RWA(self.H_mw_qubit_2, rabi_f*(np.pi), phase, freq-self.f_qubit2, t_start, t_stop)
				# if you want not to do the RWA, DO: (not if you frequency is high, you will need a lot of simulation points!)
				# self.solver_obj.add_H1_MW_RF_RWA(self.H_mw_qubit_1, rabi_f/(2*np.pi), -phase, -freq-self.f_qubit1, t_start, t_stop)
				# self.solver_obj.add_H1_MW_RF_RWA(self.H_mw_qubit_2, rabi_f/(2*np.pi), -phase, -freq-self.f_qubit2, t_start, t_stop)
			else:
				# Add microwave oject, where the gaussian is defined.
				mw_obj_1 = ME.microwave_RWA()
				mw_obj_1.init(rabi_f*(np.pi), phase, freq-self.f_qubit1, t_start, t_stop)
				mw_obj_1.add_gauss_mod(sigma_Gauss)
				mw_obj_2 = ME.microwave_RWA()
				mw_obj_2.init(rabi_f*(np.pi), phase, freq-self.f_qubit2, t_start, t_stop)
				mw_obj_2.add_gauss_mod(sigma_Gauss)
				self.solver_obj.add_H1_MW_RF_obj(self.H_mw_qubit_1, mw_obj_1)
				self.solver_obj.add_H1_MW_RF_obj(self.H_mw_qubit_2, mw_obj_2)
		else:
			MW_obj_1 = ME.microwave_pulse()
			MW_obj_1.init_normal(rabi_f*np.pi, phase, freq-self.f_qubit1, t_start, t_stop, self.H_mw_qubit_1 + self.H_mw_qubit_1.T)
			self.solver_obj.add_H1_MW_RF_obj(MW_obj_1)
			MW_obj_2 = ME.microwave_pulse()
			MW_obj_2.init_normal(rabi_f*np.pi, phase, freq-self.f_qubit2, t_start, t_stop, self.H_mw_qubit_2 + self.H_mw_qubit_2.T)
			self.solver_obj.add_H1_MW_RF_obj(MW_obj_2)

	def plot_pulse(self, mat, my_filter):
		pulse = ME.test_pulse()
		t_tot = (mat[-1,0] - mat[0,0])
		t_0  = mat[0,0] - t_tot*.3
		t_e  = mat[-1,0] + t_tot
		pulse.init(mat, my_filter)
		pulse.plot_pulse(t_0, t_e, 1e4)	

	def awg_pulse(self, amp, t_start, t_stop, bandwidth=0, plot=0):
		mat = np.zeros([4,2])
		mat[:2,0] = t_start
		mat[2:,0] = t_stop
		mat[1:3,1] = amp

		my_filter = [['Butt', 1, 300e6], ['Butt', 2, 380e6]]
		if filtering == False:
			my_filter = []

		# simple detuning pulse.
		self.solver_obj.add_H1_AWG(mat, -(self.H_charg1 - self.H_charg2)*(2*np.pi), my_filter)

		if plot == 1: 
			self.plot_pulse(mat, my_filter)
			plt.show()

	def awg_pulse_tc(self, amp, t_start, t_stop, filtering = True, plot = 0):
		# tunnen couplings pulse
		mat = np.zeros([4,2])
		mat[:2,0] = t_start
		mat[2:,0] = t_stop
		mat[1:3,1] = amp

		my_filter = [['Butt', 1, 300e6], ['Butt', 2, 380e6]]
		if filtering == False:
			my_filter = []

		self.solver_obj.add_H1_AWG(mat, self.H_tunnel*(2*np.pi),  [['Butt', 1, 300e6], ['Butt', 2, 380e6]])
		if plot == 1: 
			self.plot_pulse(mat, my_filter)
			plt.show()

	def awg_pulse_tc_custom(self, mat, bandwidth=0, plot=0):
		# tunnen couplings pulse

		self.solver_obj.add_H1_AWG(mat, self.H_tunnel*(2*np.pi),  [['Butt', 1, 300e6], ['Butt', 2, 380e6]])
		if plot == 1: 
			self.plot_pulse(mat)
			plt.show()

	def awg_pulse_B_field(self, amp1, amp2, t_start, t_stop, bandwidth=0, plot=0):
		# tunnen couplings pulse
		mat1 = np.zeros([4,2])
		mat1[:2,0] = t_start
		mat1[2:,0] = t_stop
		mat1[1:3,1] = amp1
		
		mat2 = np.zeros([4,2])
		mat2[:2,0] = t_start
		mat2[2:,0] = t_stop
		mat2[1:3,1] = amp2
		self.solver_obj.add_H1_AWG(mat1, self.H_B_field1*(2*np.pi),  [['Butt', 1, 300e6], ['Butt', 2, 380e6]])

		self.solver_obj.add_H1_AWG(mat2, self.H_B_field2*(2*np.pi),  [['Butt', 1, 300e6], ['Butt', 2, 380e6]])

		if plot == 1: 
			self.plot_pulse(mat1, my_filter)
			self.plot_pulse(mat2, my_filter)
			plt.show()

	def awg_pulse_B_field_custom_input(self, mat1, mat2, bandwidth=0, plot=0):
		# tunnen couplings pulse
		self.solver_obj.add_H1_AWG(mat1, self.H_B_field1*(2*np.pi),  [['Butt', 1, 300e6], ['Butt', 2, 380e6]])

		self.solver_obj.add_H1_AWG(mat2, self.H_B_field2*(2*np.pi),  [['Butt', 1, 300e6], ['Butt', 2, 380e6]])

	# Functions to add noise::
	def number_of_sim_for_static_noise(self, number):
		# Number of simulations to do.
		self.solver_obj.set_number_of_evalutions(number)

	def add_param_depence(self, input_matrix, i,j, function):
		# Add dependence, e.g. add magnetic fields when epsilon changes.
		self.solver_obj.add_H1_element_dep_f(input_matrix,i,j, function)

	def add_magnetic_noise(self, T2_qubit_1, T2_qubit_2):
		# 2 double as input. Just put your times
		self.solver_obj.add_magnetic_noise(self.H_B_field1, T2_qubit_1)
		self.solver_obj.add_magnetic_noise(self.H_B_field2, T2_qubit_2)

	def add_noise_object(self, magnetic_noise_object):
		# add noise that for example depends on the detuning ...
		self.solver_obj.add_noise_obj(magnetic_noise_object)

	def set_amplitude_1f_noise(self, amp,alpha=1.):
		# add 1f noise over the whole simulation.
		self.solver_obj.add_1f_noise(self.H_charg1 - self.H_charg2, amp,alpha)

	# Wrapper for calc func
	def calc_time_evolution(self, psi0, t_start,t_end,steps):
		self.len_sim = t_end-t_start
		self.solver_obj.calculate_evolution(psi0, t_start, t_end, steps)
	
	# visuals:
	def plot_pop(self):
		dd = np.array(list(basis(6,0)*basis(6,0).dag()))[:,0]
		du = np.array(list(basis(6,1)*basis(6,1).dag()))[:,0]
		ud = np.array(list(basis(6,2)*basis(6,2).dag()))[:,0]
		uu = np.array(list(basis(6,3)*basis(6,3).dag()))[:,0]
		s1 = np.array(list(basis(6,4)*basis(6,4).dag()))[:,0]
		s2 = np.array(list(basis(6,5)*basis(6,5).dag()))[:,0]
		operators = np.array([dd,du,ud,uu,s1,s2],dtype=complex) #
		label = ["dd", "du", "ud", "uu", "Sl","Sr"]
		self.solver_obj.plot_expectation(operators, label,1)

		# print(self.solver_obj.return_expectation_values(operators))
	def save_pop(self, location):
		dd = np.array(list(basis(6,0)*basis(6,0).dag()))
		du = np.array(list(basis(6,1)*basis(6,1).dag()))[:,0]
		ud = np.array(list(basis(6,2)*basis(6,2).dag()))[:,0]
		uu = np.array(list(basis(6,3)*basis(6,3).dag()))[:,0]
		s1 = np.array(list(basis(6,4)*basis(6,4).dag()))[:,0]
		s2 = np.array(list(basis(6,5)*basis(6,5).dag()))[:,0]
		operators = np.array([dd,du,ud,uu,s1,s2],dtype=complex) #
		label = ["dd", "du", "ud", "uu", "Sl","Sr"]
		expect = self.solver_obj.return_expectation_values(operators)
		times = self.solver_obj.get_times()
		data_obj = np.zeros([len(operators)+1, len(times)])

		data_obj[0]  = times
		data_obj[1:] = expect
		
		header = "Colomn 1: Time\n"
		for i in range(len(label)):
			header += "Colomn " + str(i) + ": " + label[i]+"\n"

		np.savetxt(location, data_obj.T, header=header)

	def plot_expect(self):
		XI = np.array(list(basis(6,1)*basis(6,3).dag() + basis(6,0)*basis(6,2).dag() + basis(6,2)*basis(6,0).dag() + basis(6,3)*basis(6,1).dag()))[:,0]
		IX = np.array(list(basis(6,0)*basis(6,1).dag() + basis(6,1)*basis(6,0).dag() + basis(6,2)*basis(6,3).dag() + basis(6,3)*basis(6,2).dag()))[:,0]
		XX = np.array(list(basis(6,0)*basis(6,3).dag() + basis(6,1)*basis(6,2).dag() + basis(6,2)*basis(6,1).dag() + basis(6,3)*basis(6,0).dag()))[:,0]
		ZZ = np.array(list(basis(6,0)*basis(6,0).dag() - basis(6,1)*basis(6,1).dag() - basis(6,2)*basis(6,2).dag() + basis(6,3)*basis(6,3).dag()))[:,0]
		ZI = np.array(list(basis(6,0)*basis(6,0).dag() + basis(6,1)*basis(6,1).dag() - basis(6,2)*basis(6,2).dag() - basis(6,3)*basis(6,3).dag()))[:,0]
		IZ = np.array(list(basis(6,0)*basis(6,0).dag() - basis(6,1)*basis(6,1).dag() + basis(6,2)*basis(6,2).dag() - basis(6,3)*basis(6,3).dag()))[:,0]
		YY_elem = np.array(list(tensor(sigmay(), sigmay())))[:,0]
		YY = np.zeros([6,6], dtype=np.complex)
		YY[:4,:4] = YY_elem
		operators = np.array([ZI,IZ,ZZ,XI,IX,XX,YY],dtype=complex)

		label = ["ZI", "IZ", "ZZ", "XI", "IX", "XX","YY"]
		self.solver_obj.plot_expectation(operators, label,2)

	def get_unitary(self):
		U = self.solver_obj.get_unitary()
		return U

	def get_fidelity(self, target, U = None):
		if U is None:
			U = self.get_unitary()
		U = qt.Qobj(list(U))

		# Correct for phase of singlet. We do not give a damn about it.
		target[4,4] = cmath.rect(1, cmath.phase(U[4,4]))
		target[5,5] = cmath.rect(1, cmath.phase(U[5,5]))
		target = qt.Qobj(list(target))

		return (qt.average_gate_fidelity(U,target=target))

	def get_purity(self, plot = True):
		# Get purities of the DM
		mat = self.solver_obj.get_all_density_matrices()[:,:4,:4]
		times = self.solver_obj.get_times()*1e9
		# Purity qubit 1
		trace_qubit_2 = np.zeros((mat.shape[0],2,2), dtype=np.complex)
		trace_qubit_2[:,0,0] = mat[:,0,0] +  mat[:,1,1]
		trace_qubit_2[:,0,1] = mat[:,0,2] +  mat[:,1,3]
		trace_qubit_2[:,1,0] = mat[:,2,0] +  mat[:,3,1]
		trace_qubit_2[:,1,1] = mat[:,2,2] +  mat[:,3,3]
		
		purity_qubit_1 = self.take_traces_of_matrix(self.matrix_power_list(trace_qubit_2))

		# Purity qubit 2
		trace_qubit_1 = np.zeros((mat.shape[0],2,2), dtype=np.complex)
		trace_qubit_1[:,0,0] = mat[:,0,0] +  mat[:,2,2]
		trace_qubit_1[:,0,1] = mat[:,0,1] +  mat[:,2,3]
		trace_qubit_1[:,1,0] = mat[:,1,0] +  mat[:,3,2]
		trace_qubit_1[:,1,1] = mat[:,1,1] +  mat[:,3,3]

		purity_qubit_2 = self.take_traces_of_matrix(self.matrix_power_list(trace_qubit_1))
		# Purity whole system
		purity_sys = self.take_traces_of_matrix(self.matrix_power_list(mat))

		if plot ==  True:
			plt.figure(3)
			plt.plot(times,purity_qubit_1, label="Purity qubit 1")
			plt.plot(times,purity_qubit_2, label="Purity qubit 2")
			plt.plot(times,purity_sys, label="Purity system")
			plt.xlabel('Time (ns)')
			plt.ylabel('Purity (%)')
			plt.legend()


		return purity_qubit_1, purity_qubit_2, purity_sys

	def save_purities(self,name):
		pur1, pur2, pur3 = self.get_purity(plot=False)

		# Make appropriate save format.
		pur_1 = np.empty((2,pur1.shape[0]))
		pur_1[0] = self.solver_obj.get_times()
		pur_1[1] = pur1

		pur_2 = np.empty((2,pur1.shape[0]))
		pur_2[0] = self.solver_obj.get_times()
		pur_2[1] = pur2

		pur_3 = np.empty((2,pur1.shape[0]))
		pur_3[0] = self.solver_obj.get_times()
		pur_3[1] = pur3 

		np.savetxt( name + "_purity_qubit_1.txt", pur_1.T, header='col 1: time (s)\ncol 2 : purity (%)')
		np.savetxt( name + "_purity_qubit_2.txt", pur_2.T, header='col 1: time (s)\ncol 2 : purity (%)')
		np.savetxt( name + "_purity_qubit_system.txt", pur_3.T, header='col 1: time (s)\ncol 2 : purity (%)')

	def plot_ST_bloch_sphere(self):
		mat = self.solver_obj.get_all_density_matrices()[:,1:3,1:3]
		
		k = np.linspace(0, len(mat)-1, 100, dtype=np.int)
		b=Bloch()
		b.xlabel = ['S','T']
		b.ylabel = ['S+iT', 'S-iT']
		b.zlabel = ['01','10']

		b.add_states(Qobj(list(mat[0])))
		b.add_states(Qobj(list(mat[-1])))
		x = []
		y = []
		z = []
		for i in k:
			x.append(expect(sigmax(), Qobj(list(mat[i]))))
			y.append(expect(sigmay(), Qobj(list(mat[i]))))
			z.append(expect(sigmaz(), Qobj(list(mat[i]))))
		b.add_points([x, y, z], meth='l')
		b.show()
	
	def get_density_matrix_final(self):
		return self.solver_obj.get_all_density_matrices()[-1]


# Example single qubit gate

test_single_qubit_gate = double_dot_hamiltonian(2e9,2.5e9,2e12,2.1e12,0)

test_single_qubit_gate.mw_pulse(2e9,0,2e6,0e-9,500e-9, RWA= True)
# test_single_qubit_gate.mw_pulse(2.15e9,0,5e6,0e-9,500e-9, RWA= True)

DM =	np.zeros([6,6], dtype=np.complex)
DM[1,1] = 1

test_single_qubit_gate.calc_time_evolution(DM, 0,200e-9,10000)

test_single_qubit_gate.plot_pop()
plt.show()

# # Example two qubit gate (sqwt swap based cphase)

# solver = double_dot_hamiltonian(2e9,2e9,1e12,1e12,0e1)


# solver.awg_pulse_tc(4.3e9,1e-9,5e-9)
# solver.awg_pulse_B_field(-10e6, 10e6, 10e-9, 35e-9)
# solver.awg_pulse_tc(4.3e9,40e-9,44e-9)

# dm = np.zeros([6,6], dtype=np.complex)
# dm[1,1] = 1

# solver.calc_time_evolution(dm, 0, 50e-9,25000)

# print(solver.get_unitary())

# solver.plot_pop()
# solver.plot_expect()
# solver.plot_ST_bloch_sphere()
# plt.show()


