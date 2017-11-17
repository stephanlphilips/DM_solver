import numpy as np
import scipy as sp
import c_solver.ME_solver as ME
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


        self.H_B_field1 = np.array(list(-basis(6,0)*basis(6,0).dag() - basis(6,1)*basis(6,1).dag() + basis(6,2)*basis(6,2).dag() + basis(6,3)*basis(6,3).dag()))[:,0]/2
        self.H_B_field2 = np.array(list(-basis(6,0)*basis(6,0).dag() + basis(6,1)*basis(6,1).dag()- basis(6,2)*basis(6,2).dag() + basis(6,3)*basis(6,3).dag()))[:,0]/2

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

        # add time dependent tunnelcouplings (see presentation)
        locations_1 = np.array([[1,4]],dtype=np.int32)
        locations_2 = np.array([[2,4]], dtype=np.int32)
        self.solver_obj.add_cexp_time_dep(locations_1, (self.f_qubit1-self.f_qubit2)/2)
        self.solver_obj.add_cexp_time_dep(locations_2, (self.f_qubit2-self.f_qubit1)/2)

        locations_1 = np.array([[1,5]],dtype=np.int32)
        locations_2 = np.array([[2,5]], dtype=np.int32)
        self.solver_obj.add_cexp_time_dep(locations_1, (self.f_qubit1-self.f_qubit2)/2)
        self.solver_obj.add_cexp_time_dep(locations_2, (self.f_qubit2-self.f_qubit1)/2)
    
    def return_eigen_values_vector(self, B1,B2, chargingE1, chargingE2, tunnel_coupling):
        H = B1*self.H_B_field1 + B2*self.H_B_field2 + chargingE1*self.H_charg1 + chargingE2*self.H_charg2 + tunnel_coupling*self.H_tunnel
        # H *= 2*np.pi
        return np.linalg.eig(H)

    def return_hamiltonian(self, epsilon = 0):
        return self.H_charg1*self.chargingE1 + self.H_charg2*self.chargingE2 + \
                self.H_tunnel*self.tunnel_coupling + \
                self.B_1*self.H_B_field1 + self.B_2*self.H_B_field2 + \
                self.H_charg1*epsilon - self.H_charg2*epsilon

    def return_B1(self):
        return self.H_B_field1
    def return_B2(self):
        return self.H_B_field2

    # Functions that can simulate the effect of MW signals and pulses to the sample.
    # Quasi dirty implementation of gaussian pulses.
    def mw_pulse(self, freq, phase, rabi_f, t_start, t_stop, sigma_Gauss= None):
        self.H_mw_qubit_1 = np.array(list(basis(6,0)*basis(6,2).dag() + basis(6,1)*basis(6,3).dag()))[:,0]
        self.H_mw_qubit_2 = np.array(list(basis(6,0)*basis(6,1).dag() + basis(6,2)*basis(6,3).dag()))[:,0]

        if sigma_Gauss == None:
            # Note RWA
            self.solver_obj.add_H1_MW_RF(self.H_mw_qubit_1, rabi_f*(np.pi), phase, freq-self.f_qubit1, t_start, t_stop)
            self.solver_obj.add_H1_MW_RF(self.H_mw_qubit_2, rabi_f*(np.pi), phase, freq-self.f_qubit2, t_start, t_stop)
            # if you want not to do the RWA, DO: (not if you frequency is high, you will need a lot of simulation points!)
            # self.solver_obj.add_H1_MW_RF(self.H_mw_qubit_1, rabi_f/(2*np.pi), -phase, -freq-self.f_qubit1, t_start, t_stop)
            # self.solver_obj.add_H1_MW_RF(self.H_mw_qubit_2, rabi_f/(2*np.pi), -phase, -freq-self.f_qubit2, t_start, t_stop)
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
            
    def awg_pulse(self, amp, t_start, t_stop, skew, plot=0):
        # simple detuning pulse.
        self.solver_obj.add_H1_AWG(self.H_charg1 - self.H_charg2, -amp*2*np.pi, skew,t_start,t_stop)

    def awg_pulse_tc(self, amp, t_start, t_stop, skew, plot=0):
        # tunnen couplings pulse
        self.solver_obj.add_H1_AWG(self.H_tunnel,amp*2*np.pi, skew,t_start,t_stop)

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
        dd = np.array(list(basis(6,0)*basis(6,0).dag()))[:,0]
        du = np.array(list(basis(6,1)*basis(6,1).dag()))[:,0]
        ud = np.array(list(basis(6,2)*basis(6,2).dag()))[:,0]
        uu = np.array(list(basis(6,3)*basis(6,3).dag()))[:,0]
        s1 = np.array(list(basis(6,4)*basis(6,4).dag()))[:,0]
        s2 = np.array(list(basis(6,5)*basis(6,5).dag()))[:,0]
        operators = np.array([dd,du,ud,uu,s1,s2],dtype=complex) #
        label = ["dd", "du", "ud", "uu", "Sl","Sr"]
        expect = self.solver_obj.get_expectation(operators)
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

    def matrix_power_list(self, matrix):
        # Just squares a list of matrices...
        mat_cpy = np.empty(matrix.shape, dtype=np.complex)
        for i in range(matrix.shape[0]):
            mat_cpy[i] = np.dot(np.conj(matrix[i].T),matrix[i])

        return mat_cpy

    def take_traces_of_matrix(self, matrix):
        # Takes traces of matrices.
        tr = np.zeros(matrix.shape[0])

        for i in range(tr.shape[0]):
            tr[i] = np.abs(np.trace(matrix[i]))
        return tr 

    def plot_ST_bloch_sphere(self):
        mat = self.solver_obj.get_all_density_matrices()[:,1:3,1:3]
        
        k = np.linspace(0, len(mat)-1, 100, dtype=np.int)
        b=Bloch()
        b.xlabel = ['S','T']
        b.ylabel = ['S+iT', 'S-iT']
        b.zlabel = ['01','10']
        for i in k:
            b.add_states(Qobj(list(mat[i])))
        b.show()
    
    def get_density_matrix_final(self):
        return self.solver_obj.get_all_density_matrices()[-1]

    def clear(self):
        self.solver_obj.clear()
