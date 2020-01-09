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


class noise_calibration():
    """docstring for noise calibration"""
    def __init__(self, f1, f2,  j_max):
        #Time of the simulation in nanoseconds
        self.runs=int(500)
        self.total_time=15000
        self.total_time_exch = 2000
        self.f_qubit1 = f1
        self.f_qubit2 = f2
        self.delta_z = (f1-f2)*1e9
        self.exchange_dc =[0.,0.2*1e6,0.5*1e6,1.*1e6,j_max*1e9]
        
        #Set up Zeeman Hamiltonian
        self.H_zeeman = np.zeros([4,4],dtype=np.complex)
        self.H_zeeman[1,1]=1/2
        self.H_zeeman[2,2]=-1/2
        
        #Set up Exchange Hamiltonian
        self.H_heisenberg = np.zeros([4,4],dtype=np.complex)
        self.H_heisenberg[1,1]=-1/2
        self.H_heisenberg[2,2]=-1/2
        self.H_heisenberg[1,2]=1/2
        self.H_heisenberg[2,1]=1/2
        
        #Set basis transformation matrix for exchange settings
        #self.trafo = list()
        #for exch in self.exchange_dc:
        #    eig_val, eig_vec = sp.linalg.eigh(self.H_zeeman*self.delta_z + self.H_heisenberg * exch)
        #    self.trafo.append(np.transpose(np.array(eig_vec)))
        
        #Set initial states for exchange settings
        self.init_states_ST = list()
        self.init_states_Bell = list()
        for exch in self.exchange_dc:
            eig_val, eig_vec = sp.linalg.eigh(np.diag(1e10*np.array([1.,0.,0.,-1.]))+self.H_zeeman*self.delta_z + self.H_heisenberg * exch)
            temp_vec=(qt.Qobj(eig_vec[1])+qt.Qobj(eig_vec[2]))/np.sqrt(2)
            self.init_states_ST.append(np.array((temp_vec*temp_vec.dag()).full()))
            temp_vec=(qt.Qobj(eig_vec[0])+qt.Qobj(eig_vec[1]))/np.sqrt(2)
            self.init_states_Bell.append(np.array((temp_vec*temp_vec.dag()).full()))
        
        
        # Set up noise parameters
        oneoverfnoise=lambda omega: 1/2/np.pi/omega
        self.T2sQ1 = 1.3*1e-5 #1.7*1e-6
        self.T2sQ2 = 1.1*1e-5 #1.2*1e-6
        self.j_noise = 1.*1e9
        self.T2sJ = 1.*1e-7  # T2s is suppressed by magnetic field gradient
        
        # Is Ramsey possible to map full decay of charge noise in exchange
        
        # Set up noise Hamiltonians
        self.H_zeeman_Q1 = np.zeros([4,4],dtype=np.complex)
        self.H_zeeman_Q1[0,0]=1/2
        self.H_zeeman_Q1[1,1]=1/2
        self.H_zeeman_Q1[2,2]=-1/2
        self.H_zeeman_Q1[3,3]=-1/2
        
        self.H_zeeman_Q2 = np.zeros([4,4],dtype=np.complex)
        self.H_zeeman_Q2[0,0]=1/2
        self.H_zeeman_Q2[1,1]=-1/2
        self.H_zeeman_Q2[2,2]=1/2
        self.H_zeeman_Q2[3,3]=-1/2
        
        
        
        
        self.list_exp_Q1 = list()
        self.list_exp_Q2 = list()
        self.list_exp_J_ST = list()
        self.list_exp_J_Bell = list()
        for exch_it in range(len(self.exchange_dc)):
            # Add noise to Hamiltonian
            self.solver_obj = DM.DM_solver()
            self.solver_obj.add_H0(2*np.pi*self.H_zeeman,self.delta_z)
            self.solver_obj.add_H0(2*np.pi*self.H_heisenberg,self.exchange_dc[exch_it])
            
            self.solver_obj.add_noise_static(2*np.pi*self.H_zeeman_Q1,self.T2sQ1)
            self.solver_obj.add_noise_static(2*np.pi*self.H_zeeman_Q2,self.T2sQ2)
            #self.solver_obj.add_noise_static(2*np.pi*self.H_heisenberg,self.T2sJ/(self.exchange_dc[exch_it]+1e-5)*self.exchange_dc[-1])
            self.solver_obj.add_noise_generic(2*np.pi*self.H_heisenberg,oneoverfnoise,self.j_noise*(self.exchange_dc[exch_it]/self.exchange_dc[-1]))
            
            # Compute time evolution
            self.init = np.zeros([4,4], dtype=np.complex)
            self.init[0,0] = 1/2
            self.init[2,0] = 1/2
            self.init[0,2] = 1/2
            self.init[2,2] = 1/2
            self.solver_obj.calculate_evolution(self.init,self.total_time,self.total_time*10,self.runs)
            self.list_exp_Q1.append(self.solver_obj.return_expectation_values())
            
            self.init = np.zeros([4,4], dtype=np.complex)
            self.init[0,0] = 1/2
            self.init[1,0] = 1/2
            self.init[0,1] = 1/2
            self.init[1,1] = 1/2
            self.solver_obj.calculate_evolution(self.init,self.total_time,self.total_time*10,self.runs)
            self.list_exp_Q2.append(self.solver_obj.return_expectation_values())
            
            self.init = self.init_states_ST[exch_it]
            self.solver_obj.calculate_evolution(self.init,self.total_time_exch,self.total_time_exch*10,self.runs)
            self.list_exp_J_ST.append(self.solver_obj.return_expectation_values())
            
            self.init = self.init_states_Bell[exch_it]
            self.solver_obj.calculate_evolution(self.init,self.total_time_exch,self.total_time_exch*10,self.runs)
            self.list_exp_J_Bell.append(self.solver_obj.return_expectation_values())
            
            
        expect , time, label = self.list_exp_Q1[0]
        plt.plot(time, expect[3])
        plt.show()
        
        expect , time, label = self.list_exp_Q2[0]
        plt.plot(time, expect[4])
        plt.show()
        
        expect , time, label = self.list_exp_J_ST[-1]
        plt.plot(time, expect[7])
        plt.show()
        
        expect , time, label = self.list_exp_J_Bell[-1]
        plt.plot(time, expect[8])
        plt.show()
    
class noise_beating():
    """docstring for noise calibration"""
    def __init__(self, f1):
        #Time of the simulation in nanoseconds
        self.runs=int(500)
        self.total_time=20000
        self.f_qubit1 = f1
        
        #Set up Zeeman Hamiltonian
        self.H_zeeman = np.zeros([2,2],dtype=np.complex)
        self.H_zeeman[0,0]=1/2
        self.H_zeeman[1,1]=-1/2
        
        
        #Set basis transformation matrix for exchange settings
        #self.trafo = list()
        #for exch in self.exchange_dc:
        #    eig_val, eig_vec = sp.linalg.eigh(self.H_zeeman*self.delta_z + self.H_heisenberg * exch)
        #    self.trafo.append(np.transpose(np.array(eig_vec)))
       
        # Set up noise parameters
        oneoverfnoise=lambda omega: 1/2/np.pi/omega
        twolevelfluc=lambda omega: 1/2/np.pi/(10**10+omega**2)
        self.T2sQ1 = 1.3*1e-5 #1.7*1e-6
        self.T2sQ2 = 1.1*1e-5 #1.2*1e-6
        self.j_noise = 1.*1e9
        self.T2sJ = 1.*1e-7  # T2s is suppressed by magnetic field gradient
        
        # Is Ramsey possible to map full decay of charge noise in exchange
        
        # Set up noise Hamiltonians
        self.H_zeeman_Q1 = np.zeros([2,2],dtype=np.complex)
        self.H_zeeman_Q1[0,0]=1/2
        self.H_zeeman_Q1[1,1]=-1/2
        
        
        
        self.list_exp_Qstatic = list()
        self.list_exp_Q1overf = list()
        self.list_exp_Q2levelFluc = list()
            # Add noise to Hamiltonian
        self.solver_obj = DM.DM_solver()
        self.solver_obj.add_H0(2*np.pi*self.H_zeeman,self.f_qubit1)
            
        self.solver_obj.add_noise_static(2*np.pi*self.H_zeeman,self.T2sQ1)
        
        self.solver_obj.add_noise_generic(2*np.pi*self.H_zeeman_Q1,twolevelfluc,self.T2sQ1)
            
        # Compute time evolution
        self.init = np.zeros([2,2], dtype=np.complex)
        self.init[0,0] = 1/2
        self.init[1,0] = 1/2
        self.init[0,1] = 1/2
        self.init[1,1] = 1/2
        self.solver_obj.calculate_evolution(self.init,self.total_time,self.total_time*10,self.runs)
        self.list_exp_Qstatic = self.solver_obj.return_expectation_values_general([self.init])
        
        expect , time = self.list_exp_Qstatic
        plt.plot(time, expect[0])
        plt.show()
        
f1 = 7.8
f2 = 7.6
exch= 0.023021


#noise_calibration(f1, f2,exch)
noise_beating(f1)
#print(qt.to_super(cphase).data)
