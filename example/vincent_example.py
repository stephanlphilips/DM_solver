import numpy as np
import scipy as sp

import sys
import  c_solver.ME_solver as ME

import matplotlib.pyplot as plt
import qutip as qt
import cmath


class two_qubit_sim_full():
	"""docstring for two_qubit_sim -- no RWA used"""
	def __init__(self, f1, f2):

		self.f_qubit1 = f1
		self.f_qubit2 = f2

		# define hamiltonian for two qubits (zeeman on qubit 1, zeeman on qubit 2, exchange between qubits) -- convert from qutip to numpy (brackets)
		self.B1 = qt.tensor(qt.sigmaz()/2, qt.qeye(2))[:,:]
		self.B2 = qt.tensor(qt.qeye(2), qt.sigmaz()/2)[:,:]
		# exchange energy, a thing for later ;_
		self.J  = (qt.tensor(qt.sigmax(), qt.sigmax()) + qt.tensor(qt.sigmay(), qt.sigmay()) + qt.tensor(qt.sigmaz(), qt.sigmaz()))[:,:]/4

		self.my_Hamiltonian = 2*np.pi*(f1*self.B1 + f2*self.B2)
		# Create the solver
		self.solver_obj = ME.VonNeumann(4)
		# Add the init hamiltonian
		self.solver_obj.add_H0(self.my_Hamiltonian)

	def mw_pulse(self, freq, phase, rabi_f, t_start, t_stop):

		self.H_mw_qubit_1 = np.array(list(qt.basis(4,0)*qt.basis(4,2).dag() + qt.basis(4,1)*qt.basis(4,3).dag()))[:,0]
		self.H_mw_qubit_2 = np.array(list(qt.basis(4,0)*qt.basis(4,1).dag() + qt.basis(4,2)*qt.basis(4,3).dag()))[:,0]

		MW_obj_1 = ME.microwave_pulse()
		MW_obj_1.init_normal(rabi_f*np.pi, phase, freq, t_start, t_stop,self.H_mw_qubit_1 + self.H_mw_qubit_1.T)
		self.solver_obj.add_H1_MW_RF_obj(MW_obj_1)

		MW_obj_2 = ME.microwave_pulse()
		MW_obj_2.init_normal(rabi_f*np.pi, phase, freq, t_start, t_stop,self.H_mw_qubit_2 + self.H_mw_qubit_2.T)
		self.solver_obj.add_H1_MW_RF_obj(MW_obj_2)

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
		dd = np.array(list(qt.basis(4,0)*qt.basis(4,0).dag()))[:,0]
		du = np.array(list(qt.basis(4,1)*qt.basis(4,1).dag()))[:,0]
		ud = np.array(list(qt.basis(4,2)*qt.basis(4,2).dag()))[:,0]
		uu = np.array(list(qt.basis(4,3)*qt.basis(4,3).dag()))[:,0]
		operators = np.array([dd,du,ud,uu],dtype=complex) #
		label = ["dd", "du", "ud", "uu"]
		self.solver_obj.plot_expectation(operators, label,1)

	def plot_expect(self):
		XI = np.array(list(qt.basis(4,1)*qt.basis(4,3).dag() + qt.basis(4,0)*qt.basis(4,2).dag() + qt.basis(4,2)*qt.basis(4,0).dag() + qt.basis(4,3)*qt.basis(4,1).dag()))[:,0]
		IX = np.array(list(qt.basis(4,0)*qt.basis(4,1).dag() + qt.basis(4,1)*qt.basis(4,0).dag() + qt.basis(4,2)*qt.basis(4,3).dag() + qt.basis(4,3)*qt.basis(4,2).dag()))[:,0]
		XX = np.array(list(qt.basis(4,0)*qt.basis(4,3).dag() + qt.basis(4,1)*qt.basis(4,2).dag() + qt.basis(4,2)*qt.basis(4,1).dag() + qt.basis(4,3)*qt.basis(4,0).dag()))[:,0]
		ZZ = np.array(list(qt.basis(4,0)*qt.basis(4,0).dag() - qt.basis(4,1)*qt.basis(4,1).dag() - qt.basis(4,2)*qt.basis(4,2).dag() + qt.basis(4,3)*qt.basis(4,3).dag()))[:,0]
		ZI = np.array(list(qt.basis(4,0)*qt.basis(4,0).dag() + qt.basis(4,1)*qt.basis(4,1).dag() - qt.basis(4,2)*qt.basis(4,2).dag() - qt.basis(4,3)*qt.basis(4,3).dag()))[:,0]
		IZ = np.array(list(qt.basis(4,0)*qt.basis(4,0).dag() - qt.basis(4,1)*qt.basis(4,1).dag() + qt.basis(4,2)*qt.basis(4,2).dag() - qt.basis(4,3)*qt.basis(4,3).dag()))[:,0]
		YY = qt.tensor(qt.sigmay(), qt.sigmay())[:,:]

		operators = np.array([ZI,IZ,ZZ,XI,IX,XX,YY],dtype=complex)

		label = ["ZI", "IZ", "ZZ", "XI", "IX", "XX","YY"]
		self.solver_obj.plot_expectation(operators, label,2)


	def get_expect(self):
		XI = np.array(list(qt.basis(4,1)*qt.basis(4,3).dag() + qt.basis(4,0)*qt.basis(4,2).dag() + qt.basis(4,2)*qt.basis(4,0).dag() + qt.basis(4,3)*qt.basis(4,1).dag()))[:,0]
		IX = np.array(list(qt.basis(4,0)*qt.basis(4,1).dag() + qt.basis(4,1)*qt.basis(4,0).dag() + qt.basis(4,2)*qt.basis(4,3).dag() + qt.basis(4,3)*qt.basis(4,2).dag()))[:,0]
		return self.solver_obj.return_expectation_values(np.array([XI,IX],dtype=complex))

f1 = 1.0e9
f2 = 1.1e9
sim = two_qubit_sim_full(f1, f2)

sim_time = 500e-9
n_steps = 2**18
sim.mw_pulse(freq = 0.9e9, phase = 0, rabi_f = 10e6, t_start = 0, t_stop = sim_time)
# sim.mw_pulse(freq = f2, phase = 0, rabi_f = 10e6, t_start = 100e-9, t_stop = 150e-9)
# reverse.
# sim.mw_pulse(freq = f1, phase = np.pi, rabi_f = 10e6, t_start = 200e-9, t_stop = 250e-9)
# sim.mw_pulse(freq = f2, phase = np.pi, rabi_f = 10e6, t_start = 200e-9, t_stop = 250e-9)
DM = np.zeros([4,4], dtype=np.complex)
DM[0,0] = 1
# DM[2,0] = 0.5
# DM[0,2] = 0.5
# DM[2,2] = 0.5
# make sure to take enough time steps (~100 step per period seems to give relatively accurate results, here 1.5Ghz, so one period is ~600ps, so if you take setpsize of 5ps, for 250ns, 5e4 points needed.)
# sim.calc_time_evolution(DM, 0, sim_time, n_steps)
# sim.plot_pop()
# sim.plot_expect()
# t = sim.get_expect()
import numpy as np
import matplotlib.pyplot as plt

# delta_t = sim_time/n_steps

# ft1 = np.fft.fft(t[0])
# ft2 = np.fft.fft(t[1])
# frq = np.fft.fftfreq(len(t[0]), delta_t )
# plt.plot(frq, ft1)
# plt.plot(frq, ft2)
# # plt.plot(t[1])
# plt.show()

theta = np.arctan(10/100)
print(theta)
print(np.cos(theta))
print(np.sin(theta))