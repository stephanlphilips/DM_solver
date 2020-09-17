#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:47:42 2019

@author: mruss
"""
import numpy as np
import scipy as sp

import sys
import  c_solver.DM_solver as DM

import matplotlib.pyplot as plt
from qutip import *
import qutip as qt
import cmath
import c_solver.pulse_generation.pulse_generic as pgen


class cphase():
    """docstring for two_qubit_simumlation of cphase gate in rotating frame of global Magnetic field"""
    def __init__(self, list_frequency, list_exchange, pos_q1 = 0, pos_q2 = 1):
        
        # Set up Hamiltonian
        self.position_Q1= pos_q1
        self.position_Q2 = pos_q2
        
        self.f_qubits = list_frequency
        self.ex_qubits = list_exchange
        
        
        self.number_qubits = np.count_nonzero(self.f_qubits)
        self.dimension = 2**self.number_qubits
        
        #Time of the simulation in nanoseconds
        
        self.ramp_time=10
        self.total_time=(1./j_max/2+self.ramp_time)
        
        
        self.delta_z = (f1-f2)*1e9
        self.exchange_dc =j_max*1e9

        self.H_zeeman = np.zeros([self.dimension,self.dimension],dtype=np.complex)
        self.H_zeeman[1,1]=1/2
        self.H_zeeman[2,2]=-1/2
        
        self.H_heisenberg = np.zeros([4,4],dtype=np.complex)
        self.H_heisenberg[1,1]=-1/2
        self.H_heisenberg[2,2]=-1/2
        self.H_heisenberg[1,2]=1/2
        self.H_heisenberg[2,1]=1/2
        
        oneoverfnoise=lambda omega: 1/2/np.pi/omega
        #whitenoise=lambda omega: 0.*omega + 1
        self.init = np.zeros([4,4], dtype=np.complex)
        self.init[2,2] = 1
        #self.init[2,2] = 1/2
        #self.init[1,2] = 1/2
        #self.init[2,1] = 1/2
        self.pulseDummy=pgen.pulse()
        
        def add_dc_exchange_pulse(self, t_ramp, t_start, t_stop):
            delay=1
            def enevlope_fun(time):
                return self.exchange_dc*(np.arctan(3)+np.arctan(6*(time-delay/2)/delay))/(2*np.arctan(3))
            def enevlope_fun_ss(time):
                return self.exchange_dc*(np.arctan(3)+np.arctan(6*(1-time-delay/2)/delay))/(2*np.arctan(3))
            #self.pulseDummy.add_ramp(t_start,(t_start+t_ramp),self.exchange_dc)
            self.pulseDummy.add_function(t_start,(t_start+t_ramp),enevlope_fun)
            self.pulseDummy.add_block((t_start+t_ramp),(t_stop-t_ramp),self.exchange_dc)
            #self.pulseDummy.add_ramp_ss((t_stop-t_ramp),t_stop,self.exchange_dc,0)
            self.pulseDummy.add_function((t_stop-t_ramp),t_stop,enevlope_fun_ss)
            
            
            self.solver_obj.add_H1(2*np.pi*self.H_heisenberg,self.pulseDummy)



		# Create the solver
		#self.solver_obj = DM.calculate_evolution(4)
		# Add the init hamiltonian
        self.solver_obj = DM.DM_solver()
        self.solver_obj.add_H0(2*np.pi*self.H_zeeman,self.delta_z)
        #self.solver_obj.add_noise_static(self.H_zeeman,18*1e6)
        #self.solver_obj.add_noise_generic(self.H_heisenberg,oneoverfnoise,self.exchange_dc/100)
        add_dc_exchange_pulse(self,self.ramp_time,0.,self.total_time)
        
        #self.pulseDummy.add_ramp(0,9.4,self.exchange_dc)
        #self.pulseDummy.add_block(9.4,50-9.4,self.exchange_dc)
        #self.pulseDummy.add_ramp_ss(50-9.4,50,self.exchange_dc,0)
        
        #self.solver_obj.add_H1(self.H_heisenberg,self.pulseDummy)
        
        self.solver_obj.calculate_evolution(self.init,self.total_time,50000,10)
        self.solver_obj.plot_pop()
        #print(2*np.pi*self.delta_z*(self.total_time*1e-9))
        #print(2*np.pi*np.sum(self.pulseDummy.get_pulse(self.total_time,1e11))*1e-11)
        #print(qt.Qobj(self.solver_obj.return_final_unitary()))
        
        def get_unitary_gate_fidelity(self, U = None):
            """
            returns unitary gate fidelity
            Args
                runs (str/tuple) : number of runs to compute the average gate fidelity
            """
            self.solver_obj.calculate_evolution(self.init,self.total_time,50000,1)
            U = qt.Qobj(self.solver_obj.get_unitary()[0])

            target = qt.Qobj(sp.linalg.expm(-np.pi/4.*1j*sp.sparse.diags([0.,-1.,-1.,0.]).todense()))
            temp_phase= self.delta_z*(self.total_time*1e-9)
            SQphase= qt.Qobj(sp.linalg.expm(-2*np.pi*1j*temp_phase*self.H_zeeman))
            fidelity = qt.average_gate_fidelity(U,target*SQphase)
            
            return fidelity
        
        def get_average_gate_fidelity(self, runs = 500, target = None):
            """
            returns average gate fidelity
            Args
                runs (int) : number of runs to compute the average gate fidelity
                target (4x4 numpy array) : target unitary to compute fidelity
            """
            
            self.solver_obj.calculate_evolution(self.init,self.total_time,10000,1)
            U_ideal = self.solver_obj.get_unitary()[0]
            self.solver_obj.add_noise_static(2*np.pi*self.H_zeeman,18*1e-6)
            self.solver_obj.add_noise_generic(2*np.pi*self.H_heisenberg,oneoverfnoise,self.exchange_dc/100)
            self.solver_obj.calculate_evolution(self.init,self.total_time,10000,int(runs))
            
            U_list = self.solver_obj.get_unitary()
            

            
            # Calculate the averaged super operator in the Lioville superoperator form using column convention
            basis = [qt.basis(4,it) for it in range(4)]
            superoperator_basis = [basis_it1*basis_it2.dag() for basis_it2 in basis for basis_it1 in basis]
            averaged_map = np.zeros([16,16],dtype=np.complex)
            for u in U_list:
                temp_U=qt.Qobj(u)
                output_density = list()
                for it in range(len(superoperator_basis)):
                    temp_vec=np.array(qt.operator_to_vector(temp_U*superoperator_basis[it]*temp_U.dag()/float(runs)).full()).flatten()
                    output_density.append(np.array(temp_vec))
                averaged_map = np.add(averaged_map,np.array(output_density))
            
            # Define the target unitary operation
            if target is None:
                target = sp.linalg.expm(-np.pi/2.*1j*sp.sparse.diags([0.,-1.,-1.,0.]).todense())
            
            
            # get phase from optimizing noiseless unitary evolution
            def to_minimize_fidelity(theta):
                temp_z_gate = np.matmul(sp.linalg.expm(-2*np.pi*1j*theta*self.H_zeeman),U_ideal)
                temp_m = np.matmul(sp.conjugate(sp.transpose(target)),temp_z_gate)
                return np.real(1.-(sp.trace(np.matmul(temp_m,sp.conjugate(sp.transpose(temp_m))))+np.abs(sp.trace(temp_m))**2.)/20.)
            
            ideal_phase = sp.optimize.minimize(to_minimize_fidelity, [self.delta_z*(self.total_time*1e-9)], method='Nelder-Mead', tol=1e-6).x[0]
            
            target = np.matmul(sp.linalg.expm(2*np.pi*1j*ideal_phase*self.H_zeeman),target)
            
            
            
            # Change the shape of the averaged super operator to match the definitions used in QuTip (row convention)
            target = qt.Qobj(target)
            averaged_map = qt.Qobj(averaged_map).trans()
            averaged_map._type = 'super'
            averaged_map.dims = [[[4], [4]], [[4], [4]]]
            averaged_map.superrep  = qt.to_super(target).superrep
            
            # Calculate the average gate fidelity of the superoperator with the target unitary gate
            
            fidelity = qt.average_gate_fidelity(averaged_map,target)
            return fidelity
        
        def show_pulse_sequence(self):
            t, v  =self.pulseDummy.get_pulse(self.total_time,1e11)
            plt.plot(t,v)
            plt.xlabel('time (ns)')
            plt.ylabel('amplitude (a.u.)')
            plt.show()
        
        #print(self.solver_obj.get_unitary())
        print((1-get_average_gate_fidelity(self))*100)
    
    
    


f1 = 7.8
f2 = 7.6
exch= 0.023021


two_qubit_cphase(f1, f2,exch)
cpn = np.zeros([4,4],dtype=np.complex)
cpn[0,0]=1
cpn[1,1]=1j
cpn[2,2]=1j
cpn[3,3]=1
cphase=qt.Qobj(cpn)
#print(qt.to_super(cphase).data)
