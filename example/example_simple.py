import numpy as np
import scipy as sp

import sys
import  c_solver.ME_solver as ME

import matplotlib.pyplot as plt
from qutip import *
import qutip as qt
import cmath


class two_qubit_sim():
	"""docstring for two_qubit_sim -- no RWA used"""
	def __init__(self, f1, f2):

		self.f_qubit1 = f1
		self.f_qubit2 = f2

		# define hamiltonian for two qubits (zeeman on qubit 1, zeeman on qubit 2, exchange between qubits) -- convert from qutip to numpy (brackets)
		self.B1 = qt.tensor(qt.sigmaz()/2, qt.qeye(2))[:,:]
		self.B2 = qt.tensor(qt.qeye(2), qt.sigmaz()/2)[:,:]
		self.J  = (qt.tensor(qt.sigmax(), qt.sigmax()) + qt.tensor(qt.sigmay(), qt.sigmay()) + qt.tensor(qt.sigmaz(), qt.sigmaz()))[:,:]/4

		self.my_Hamiltonian = 2*np.pi*(f1*self.B1 + f2*self.B2)
		# Create the solver
		self.solver_obj = ME.VonNeumann(4)
		# Add the init hamiltonian
		self.solver_obj.add_H0(self.my_Hamiltonian)

	def mw_pulse(self, freq, phase, rabi_f, t_start, t_stop):

		self.H_mw_qubit_1 = np.array(list(basis(4,0)*basis(4,2).dag() + basis(4,1)*basis(4,3).dag()))[:,0]
		self.H_mw_qubit_2 = np.array(list(basis(4,0)*basis(4,1).dag() + basis(4,2)*basis(4,3).dag()))[:,0]

		self.solver_obj.add_H1_MW_RF_RWA(self.H_mw_qubit_1, rabi_f*(np.pi), phase, freq, t_start, t_stop)
		self.solver_obj.add_H1_MW_RF_RWA(self.H_mw_qubit_2, rabi_f*(np.pi), phase, freq, t_start, t_stop)
		# # if you want not to do the RWA, DO: (not if you frequency is high, you will need a lot of simulation points!)
		self.solver_obj.add_H1_MW_RF_RWA(self.H_mw_qubit_1, rabi_f*(np.pi), -phase, -freq, t_start, t_stop)
		self.solver_obj.add_H1_MW_RF_RWA(self.H_mw_qubit_2, rabi_f*(np.pi), -phase, -freq, t_start, t_stop)

	def calc_time_evolution(self, psi0, t_start,t_end,steps):
		self.solver_obj.calculate_evolution(psi0, t_start, t_end, steps)

	def RF_exchange(self, rabi, freq, t_start, t_stop, phase):
		MW_obj_1 = ME.microwave_pulse()
		MW_obj_1.init_normal(rabi*2*np.pi, phase, freq, t_start, t_stop,self.J)
		self.solver_obj.add_H1_MW_RF_obj(MW_obj_1)

	def pulsed_exchange(self, amp, t_start, t_stop):
		mat1 = np.zeros([4,2])
		mat1[:2,0] = t_start
		mat1[2:,0] = t_stop
		mat1[1:3,1] = amp*2*np.pi

		self.solver_obj.add_H1_AWG(mat1, self.J)

	def get_unitary(self):
		U = self.solver_obj.get_unitary()
		return U

	def get_fidelity(self, target, U = None):
		if U is None:
			U = self.get_unitary()
		U = qt.Qobj(list(U))

		# Correct for phase of singlet. We do not give a damn about it.
		# target[4,4] = cmath.rect(1, cmath.phase(U[4,4]))
		# target[5,5] = cmath.rect(1, cmath.phase(U[5,5]))
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

B1 = 2e9
B2 = 2.100e9 
sim = two_qubit_sim(B1, B2)


# sim.mw_pulse(B1, 0, 2.5e6, 0, 100e-9)

J = 10e6
start = 50e-9
t_gate = 1/J

wait = 0
sim.RF_exchange(J, abs(B1-B2), start, start + t_gate/2, 0)

sim.pulsed_exchange(3*J/2, start, start + t_gate/2)

sim.RF_exchange(J, abs(B1-B2), start + t_gate/2, start + t_gate, 0*np.pi)

sim.pulsed_exchange(3*J/2, start + t_gate/2, start + t_gate)

# sim.pulsed_exchange(J/2, start + t_gate/2, start + t_gate)


# sim.mw_pulse(B1, np.pi/2, 2.5e6, start + t_gate + 0e-9, start + t_gate + 100e-9)


DM = np.zeros([4,4], dtype=np.complex)
DM[2,2] = 1

sim.calc_time_evolution(DM, 0, 200e-9, 40000)
# sim.plot_pop()
# sim.plot_expect()
# sim.plot_ST_bloch_sphere()
plt.show()

U = sim.get_unitary()

U_wanted = np.array([
	[-1j,0,0,0],
	[0,0,-1,0],
	[0,-1,0,0],
	[0,0,0,-1j]
	])
print(sim.get_fidelity(U_wanted))
print(U*np.e**(1j*np.pi/4))
# print(U*np.e**(1j*np.pi/4*0))
B = (np.matrix(np.angle(U.diagonal())).T)
B = np.angle(np.matrix([1,-1,-1,1])).T
print(B)
A = np.matrix([
	[-0.5,-0.5,0.25,-0.25],
	[-0.5,0.5,-0.25,-0.25],
	[0.5,-0.5,-0.25,-0.25],
	[0.5,0.5,0.25,-0.25]])
A_1 = np.linalg.inv(A)

rad_data = A_1*B
deg_data = np.degrees(rad_data)

print("ZI = ", deg_data[0,0])
print("IZ = ", deg_data[1,0])
print("ZZ = ", deg_data[2,0])
print("II = ", deg_data[3,0])

