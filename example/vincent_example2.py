import numpy as np
import scipy as sp

import sys
import  c_solver.ME_solver as ME

import matplotlib.pyplot as plt
from qutip import *
import qutip as qt
import cmath


class two_qubit_sim_full():
	"""docstring for two_qubit_sim -- no RWA used"""
	def __init__(self, f1, f2):

		self.f_qubit1 = f1
		self.f_qubit2 = f2

		self.H_mw_qubit_1 = np.array(list(basis(4,0)*basis(4,2).dag() + basis(4,1)*basis(4,3).dag()))[:,0]
		self.H_mw_qubit_2 = np.array(list(basis(4,0)*basis(4,1).dag() + basis(4,2)*basis(4,3).dag()))[:,0]

		# define hamiltonian for two qubits (zeeman on qubit 1, zeeman on qubit 2, exchange between qubits) -- convert from qutip to numpy (brackets)
		self.B1 = qt.tensor(qt.sigmaz()/2, qt.qeye(2))[:,:]
		self.B2 = qt.tensor(qt.qeye(2), qt.sigmaz()/2)[:,:]

		# transformed into rotating frame, so energy is 0
		self.my_Hamiltonian = 0*self.B1
		# Create the solver
		self.solver_obj = ME.VonNeumann(4)
		# Add the init hamiltonian
		self.solver_obj.add_H0(self.my_Hamiltonian)

	def mw_pulse(self, freq, phase, rabi_f, t_start, t_stop, RWA= True):

		# not pi/2 as half of the wave.
		self.solver_obj.add_H1_MW_RF_RWA(self.H_mw_qubit_1, rabi_f*(np.pi)/2, phase, freq-self.f_qubit1, t_start, t_stop)
		self.solver_obj.add_H1_MW_RF_RWA(self.H_mw_qubit_2, rabi_f*(np.pi)/2, phase, freq-self.f_qubit2, t_start, t_stop)
		if RWA == False:
			self.solver_obj.add_H1_MW_RF_RWA(self.H_mw_qubit_1, rabi_f*np.pi/2, -phase, -freq-self.f_qubit1, t_start, t_stop)
			self.solver_obj.add_H1_MW_RF_RWA(self.H_mw_qubit_2, rabi_f*np.pi/2, -phase, -freq-self.f_qubit2, t_start, t_stop)

	def calc_time_evolution(self, psi0, t_start,t_end,steps):
		self.solver_obj.calculate_evolution(psi0, t_start, t_end, steps)

	def get_unitary(self):
		U = self.solver_obj.get_unitary()
		return U

	def get_fidelity(self, target, U = None):
		if U is None:
			U = self.get_unitary()
		U = qt.Qobj(list(U))

		target = qt.Qobj(list(target))

		return (qt.average_gate_fidelity(U,target=target))

	# visuals:
	def plot_pop(self):
		dd = np.array(list(basis(4,0)*basis(4,0).dag()))[:,0]
		du = np.array(list(basis(4,1)*basis(4,1).dag()))[:,0]
		ud = np.array(list(basis(4,2)*basis(4,2).dag()))[:,0]
		uu = np.array(list(basis(4,3)*basis(4,3).dag()))[:,0]
		operators = np.array([dd,du,ud,uu],dtype=complex) #
		label = ["dd", "du", "ud", "uu"]
		self.solver_obj.plot_expectation(operators, label,1)

	def plot_expect(self):
		XI = np.array(list(basis(4,1)*basis(4,3).dag() + basis(4,0)*basis(4,2).dag() + basis(4,2)*basis(4,0).dag() + basis(4,3)*basis(4,1).dag()))[:,0]
		IX = np.array(list(basis(4,0)*basis(4,1).dag() + basis(4,1)*basis(4,0).dag() + basis(4,2)*basis(4,3).dag() + basis(4,3)*basis(4,2).dag()))[:,0]
		XX = np.array(list(basis(4,0)*basis(4,3).dag() + basis(4,1)*basis(4,2).dag() + basis(4,2)*basis(4,1).dag() + basis(4,3)*basis(4,0).dag()))[:,0]
		ZZ = np.array(list(basis(4,0)*basis(4,0).dag() - basis(4,1)*basis(4,1).dag() - basis(4,2)*basis(4,2).dag() + basis(4,3)*basis(4,3).dag()))[:,0]
		ZI = np.array(list(basis(4,0)*basis(4,0).dag() + basis(4,1)*basis(4,1).dag() - basis(4,2)*basis(4,2).dag() - basis(4,3)*basis(4,3).dag()))[:,0]
		IZ = np.array(list(basis(4,0)*basis(4,0).dag() - basis(4,1)*basis(4,1).dag() + basis(4,2)*basis(4,2).dag() - basis(4,3)*basis(4,3).dag()))[:,0]
		YY = tensor(sigmay(), sigmay())[:,:]

		operators = np.array([ZI,IZ,ZZ,XI,IX,XX,YY],dtype=complex)

		label = ["ZI", "IZ", "ZZ", "XI", "IX", "XX","YY"]
		self.solver_obj.plot_expectation(operators, label,2)



f1 = 1.0e9
f2 = 1.5e9
sim = two_qubit_sim_full(f1, f2)
sim.mw_pulse(freq = f1, phase = 0, rabi_f = 10e6, t_start = 0, t_stop = 50e-9, RWA= False)
sim.mw_pulse(freq = f2, phase = 0, rabi_f = 10e6, t_start = 100e-9, t_stop = 150e-9, RWA= False)
# reverse.
sim.mw_pulse(freq = f1, phase = np.pi, rabi_f = 10e6, t_start = 200e-9, t_stop = 250e-9, RWA= False)
sim.mw_pulse(freq = f2, phase = np.pi, rabi_f = 10e6, t_start = 200e-9, t_stop = 250e-9, RWA= False)
DM = np.zeros([4,4], dtype=np.complex)
DM[0,0] = 1

# make sure to take enough time steps (~100 step per period seems to give relatively accurate results, here 1.5Ghz, so one period is ~600ps, so if you take setpsize of 5ps, for 250ns, 5e4 points needed.)
sim.calc_time_evolution(DM, 0, 250e-9, 50000)
sim.plot_pop()
sim.plot_expect()
plt.show()