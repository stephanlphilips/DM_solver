#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:43:44 2021

@author: mruss
"""

import numpy as np
import scipy as sp
import pickle
import math

import sys
from datetime import datetime
import  c_solver.DM_solver as DM
import os
import time

import matplotlib.pyplot as plt
from scipy import signal
import qutip as qt
import cmath
import c_solver.pulse_generation.pulse_generic as pgen

from pathos.multiprocessing import ProcessingPool as Pool

def pulse_shape(delta_t, sample_rate = 1,indicator = 0,indicator_sub = None):
        """
        function that has window shape at the start and at the end (first 8 and last 8 ns)

        Args:
            delta_t (double) : time in ns of the pulse.
            sample_rate (double) : sampling rate of the pulse (GS/s).

        Returns:
            evelope (np.ndarray) : array of the evelope.
        """
        if indicator_sub is None:
            window = ['boxcar','blackman','blackmanharris',
                      'hann','flattop',('kaiser',14),('tukey',0.5)]
        else:
            window = ['boxcar','blackman','blackmanharris',
                      'hann','flattop',('kaiser',indicator_sub),('tukey',indicator_sub)]
        
        n_points = int(np.ceil(delta_t*sample_rate )-1.)
        envelope = np.empty([n_points], np.double)
        envelope = signal.get_window(window[indicator], n_points*10)[::10]
        compensate_height = n_points/np.sum(envelope)
        envelope = envelope*compensate_height
        return envelope


class PSB_sweep():
    """docstring for two_qubit_simumlation of cphase gate 
        in rotating frame of global Magnetic field"""
    def __init__(self,
                 set_orbital = False,
                 leverarm_tunneling = 1., 
                 leverarm_detuning = 0.02, 
                 tunnel_coupling = 2.,
                 SO_tunnel_coupling = [0.2,0.2,0.2],
                 g_q0 = [0.,0.,2.],
                 g_q1 = [0.,0.,2.],
                 delta_ep_g_q0 = [[0.,0.,0.],
                                 [0.,0.,0.],
                                 [0.,0.,0.]],
                 delta_ep_g_q1 = [[0.,0.,0.],
                                 [0.,0.,0.],
                                 [0.,0.,0.]],
                 B_q0 = [0.,0.,0.1],
                 B_q1 = [0.,0.,0.11],
                 T2star_q0 = 13., 
                 T2star_q1 = 19.,
                 quantization_axis = [0,0,1]
                 ):
        """
        Initialize two-qubit classe
        Args:
            set_SAT (Boolean) : Consider excited (valley)-orbital states
            leverarm_tunneling (double) : leverarm between barrier gate and tunneling (GHz/V)
            leverarm_detuning (double) : leverarm between plunger gate difference and detuning (GHz/V)
            tunnel_coupling (double) : tunnel coupling (in GHz)
            SO_tunnel_coupling ([double,double,double]) : tunnel coupling of spin-orbit interaction (in [GHz,GHz,GHz])
            g_q0 ([double,double,double]) : qubit g-tensor of qubit 0
            g_q1 ([double,double,double]) : qubit g-tensor of qubit 1
            delta_ep_g_q0 ([double,double,double]) : Gradient of qubit g-tensor of qubit 0
            delta_ep_g_q1 ([double,double,double]) : Gradient of qubit g-tensor of qubit 1
            B_q0 ([double,double,double]) : magnetic field vector in dot 0 (in [T,T,T])
            B_q1 ([double,double,double]) : magnetic field vector in dot 1 (in [T,T,T])
            T2star_q0 (double) : Dephasing time of qubit 1 from Ramsey in (in microseconds)
            T2star_q1 (double) : Dephasing time of qubit 2 from Ramsey in (in microseconds)
        Returns:
            
        """
        
        global exchange_fun
        global exchange_inverse
        
        def exchange_fun(rel_voltage):
            if self.set_SAT == False:
                return np.exp(2.*rel_voltage)
            else:
                return (np.sqrt(1.0+np.exp(-2.0*(rel_voltage+np.log(2.0))))
                        -np.exp(-1.0*(rel_voltage+np.log(2.0))))**2
        
        def exchange_inverse(value):
            if self.set_SAT == False:
                return np.log(value)/2.
            else:
                return np.log(np.sqrt(value)/(np.abs(1.0-value)))
            
            
        # Set attributes of class
        mu_bohr = 1.399624493*1e10
        self.mev_to_hz = 241799050402.4/(2.*np.pi)
        
        self.leverarm_detuning = leverarm_detuning
        self.leverarm_tunneling = leverarm_tunneling
        
        self.f_q0 = np.dot(np.array(g_q0),np.array(B_q0))*mu_bohr
        self.f_q1 = np.dot(np.array(g_q1),np.array(B_q1))*mu_bohr
        self.delta_f = (self.f_q1-self.f_q0)
        self.tunnel_coupling = tunnel_coupling*1e9
        self.SO_tunnel_coupling = np.array(SO_tunnel_coupling)*1e9
        
        # self.delta_f_q0 = np.array(delta_ep_g_q0)*1e9
        # self.delta_f_q1 = np.array(delta_ep_g_q0)*1e9
        
        self.dephasing_Q0 = T2star_q0*1e-6
        self.dephasing_Q1 = T2star_q1*1e-6
        
        
        # Set up low pass filter frequency
        self.lowpass_filter = 150.*1e6
        
        
        # Preset Hamiltonians
        
        self.H_zeeman_Q0 = 2.*np.pi*np.diag([0.5,0.5,-0.5,-0.5,0])
        self.H_zeeman_Q1 = 2.*np.pi*np.diag([0.5,-0.5,0.5,-0.5,0])
        
        self.H_charge = 2.*np.pi*np.diag([0,0,0,0*1j,1.])
        
        self.H_tunneling_raw = 2.*np.pi*np.array([[0., 0., 0., 0., 0.*1j],
                                                  [0., 0., 0., 0., 1.],
                                                  [0., 0., 0., 0., -1.],
                                                  [0., 0., 0., 0., 0.],
                                                  [0., 1., -1., 0., 0.]])/np.sqrt(2)
        
        self.H_SOI_tunneling_X_raw = 2.*np.pi*np.array([[0., 0., 0., 0., -1j],
                                                  [0., 0., 0., 0., 0],
                                                  [0., 0., 0., 0., 0],
                                                  [0., 0., 0., 0., 1j],
                                                  [1j, 0., 0., -1j, 0.]])/np.sqrt(2)
        
        self.H_SOI_tunneling_Y_raw = 2.*np.pi*np.array([[0., 0., 0., 0., -1.],
                                                  [0., 0., 0., 0., 0.*1j],
                                                  [0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., -1],
                                                  [-1., 0., 0., -1., 0.]])/np.sqrt(2)
        
        self.H_SOI_tunneling_Z_raw = 2.*np.pi*np.array([[0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 1j],
                                                  [0., 0., 0., 0., 1j],
                                                  [0., 0., 0., 0., 0.],
                                                  [0., -1j, -1j, 0., 0.]])/np.sqrt(2)
        
        self.H_tunneling = (self.H_tunneling_raw*self.tunnel_coupling
                            +self.H_SOI_tunneling_X_raw*self.SO_tunnel_coupling[0]
                            +self.H_SOI_tunneling_Y_raw*self.SO_tunnel_coupling[1]
                            +self.H_SOI_tunneling_Z_raw*self.SO_tunnel_coupling[2])
        
        
        self.init = np.zeros([5,5], dtype=complex)
        self.init[4,4] = 1.
        
    def initialize_pulse_sequence(self):
        
        self.detuning_offset = -1.*self.mev_to_hz
        self.total_time = 0.
        self.pulse_detuning = pgen.pulse()
    
    def add_detuning_pulse(self,t_pulse,det_start,det_stop):
        
        self.pulse_detuning.add_ramp_ss(self.total_time,
                                     self.total_time + t_pulse,
                                     self.mev_to_hz*det_start,
                                     self.mev_to_hz*det_stop)
        self.total_time = self.total_time + t_pulse
        
        print("Exchange at final point: ", (4.*self.tunnel_coupling**2
                                            /(self.mev_to_hz*det_stop))*1e-6," MHz")
    
    def add_detuning_block(self,t_pulse,det_value):
        
        self.pulse_detuning.add_block(self.total_time,
                                     self.total_time + t_pulse,
                                     self.mev_to_hz*det_value)
        self.total_time = self.total_time + t_pulse
    
    def compute_gate_sequence(self, noise_strength , runs = 1000, 
                              quasistatic_APPROX = True):
        
        self.solver_obj = DM.DM_solver()
        
        charge_noise_amp = noise_strength
        
        print("total time: ", self.total_time)
            
        self.sample_rate = int(5000.*self.total_time)
                
        self.solver_obj.add_H0(self.H_zeeman_Q0, self.f_q0)
        self.solver_obj.add_H0(self.H_zeeman_Q1, self.f_q1)
        self.solver_obj.add_H0(self.H_tunneling, 1. )
        
        # self.pulse_detuning.add_filter(self.lowpass_filter,False)
        
        self.solver_obj.add_H1(self.H_charge,self.pulse_detuning)
        
        
        
        number_runs = 1
        
        # skips averaging if noise is expected to be negligible (careful use)
        if charge_noise_amp > 1e-10:
            #set up noise spectrum
            oneoverfnoise=lambda omega: 1./2./np.pi/omega
            number_runs = runs
            self.solver_obj.add_noise_static(self.H_zeeman_Q1,
                                             self.dephasing_Q0)
            self.solver_obj.add_noise_static(self.H_zeeman_Q1,
                                             self.dephasing_Q1)
            
            if quasistatic_APPROX == True:
                n_points = 2*2**(int(np.log2(int(self.sample_rate)))+1)
                freq_lower_bound = self.sample_rate/n_points*1e9/self.total_time
                static_noise_of_spectrum_function = charge_noise_amp/np.pi*(
                    sp.integrate.quad(oneoverfnoise, 0.1*2.*np.pi, freq_lower_bound*2.*np.pi)[0]
                    )
                static_noise_of_spectrum_function_linear_transformed = (
                    1./(np.sqrt(2*static_noise_of_spectrum_function)*np.pi)
                    )
                print("dephasing time: ",static_noise_of_spectrum_function_linear_transformed)
                
                self.solver_obj.add_noise_static(
                            self.H_charge*1e9*self.leverarm_detuning,
                            static_noise_of_spectrum_function_linear_transformed
                            )
                self.solver_obj.add_noise_static(
                            self.H_tunneling*self.leverarm_tunneling,
                            static_noise_of_spectrum_function_linear_transformed
                            )
                
            else:
                
                self.solver_obj.add_noise_generic(
                            self.H_charge*1e9*self.leverarm_detuning,
                            oneoverfnoise,
                            charge_noise_amp
                            )
                
                self.solver_obj.add_noise_generic(
                            self.H_tunneling*self.leverarm_tunneling,
                            oneoverfnoise,
                            charge_noise_amp
                            )
                
            
        
        self.solver_obj.calculate_evolution(self.init,
                                            self.total_time,
                                            self.sample_rate,int(number_runs))
        
        U_list = self.solver_obj.get_unitary()
        global global_compare
        # for it in range(self.number_of_qubits):
        #         global_compare = (sp.linalg.expm(-1j*self.phase_tracker_sequence[it]
        #                                     *self.H_zeeman[it])
        #                   @ global_compare)
        self.final_DM_sequence = self.solver_obj.get_last_density_matrix()
        
        # print("U_list sequence: ",np.around(U_list[0],4) )
        
        # Calculate the averaged super operator in the Lioville 
        # superoperator form using column convention
        basis = [qt.basis(5,it) for it in range(5)]
        superoperator_basis = [basis_it1*basis_it2.dag() 
                               for basis_it2 in basis 
                               for basis_it1 in basis]
        averaged_map = np.zeros([5**2,5**2],dtype=complex)
        for u in U_list:
            temp_U = u[:,:]
            
            temp_U = qt.Qobj(temp_U)
            
            output_density = list()
            for it in range(len(superoperator_basis)):
                temp_vec=np.array(
                    qt.operator_to_vector(temp_U*superoperator_basis[it]
                                          *temp_U.dag()
                                          /float(number_runs)
                                          ).full()).flatten()
                output_density.append(np.array(temp_vec))
            averaged_map = np.add(averaged_map,np.array(output_density))
        
        # Define the target unitary operation
        
        
        target = np.identity(5, dtype=complex)
        
        # print("target: ",np.around(target,4))
        # Change the shape of the averaged super operator to match 
        # the definitions used in QuTip (row convention)
        target = qt.Qobj(target)
        
        
        
        averaged_map = qt.Qobj(averaged_map).trans()
        averaged_map._type = 'super'
        averaged_map.dims = [[[5], [5]], 
                             [[5], [5]]]
        averaged_map.superrep  = qt.to_super(target).superrep
        
        self.target_sequence = target
        self.averaged_map_sequence = averaged_map
        
    def return_expectation_values(self,op_list):
        
        return self.solver_obj.return_expectation_values_general(op_list)
   
if __name__ == '__main__':
    
    experiment = PSB_sweep(g_q0=[0.,0.,0.2],
                           g_q1=[0.,0.,0.25],
                           B_q0=[0.,0.,1e-3],
                           B_q1=[0.,0.,1e-3],
                           SO_tunnel_coupling = [0.4,0.4,0.4])
    experiment.initialize_pulse_sequence()
    experiment.add_detuning_pulse(500, -0.5, 0.5)
    experiment.compute_gate_sequence(0.,runs=1000)
    
    target_state_1 = np.zeros([5,5], dtype=complex)
    target_state_1[1,1] = 1.
    
    target_state_2 = np.zeros([5,5], dtype=complex)
    target_state_2[2,2] = 1.
    
    target_leak_1 = np.zeros([5,5], dtype=complex)
    target_leak_1[0,0] = 1.
    
    target_leak_2 = np.zeros([5,5], dtype=complex)
    target_leak_2[3,3] = 1.
    
    target_singlet = np.zeros([5,5], dtype=complex)
    target_singlet[1,1] = 0.5
    target_singlet[1,2] = -0.5
    target_singlet[2,1] = -0.5
    target_singlet[2,2] = 0.5
    
    target_triplet = np.zeros([5,5], dtype=complex)
    target_triplet[1,1] = 0.5
    target_triplet[1,2] = 0.5
    target_triplet[2,1] = 0.5
    target_triplet[2,2] = 0.5
    
    expect, times = experiment.return_expectation_values([experiment.init,
                                                         target_state_1,
                                                         target_state_2,
                                                         target_leak_1,
                                                         target_leak_2,
                                                         target_singlet,
                                                         target_triplet])
    expect_init= expect[0][::100]
    expect_target_1 = expect[1][::100]
    expect_target_2 = expect[2][::100]
    expect_leak_1 = expect[3][::100]
    expect_leak_2 = expect[4][::100]
    expect_singlet = expect[5][::100]
    expect_triplet = expect[6][::100]
    times = times[::100]
    
    plt.figure()
    plt.plot(times,expect_init,'r')
    plt.plot(times,expect_target_1,'b')
    plt.plot(times,expect_target_2,'g')
    plt.xlabel('time (ns)')
    plt.ylabel('amplitude (a.u.)')
    plt.title("target states ")
    plt.show()
    
    
    plt.figure()
    plt.plot(times,expect_init,'r')
    plt.plot(times,expect_leak_1,'b')
    plt.plot(times,expect_leak_2,'g')
    plt.xlabel('time (ns)')
    plt.ylabel('amplitude (a.u.)')
    plt.title("leakage states ")
    plt.show()
    
    
    plt.figure()
    plt.plot(times,expect_init,'r')
    plt.plot(times,expect_singlet,'b')
    plt.plot(times,expect_triplet,'g')
    plt.xlabel('time (ns)')
    plt.ylabel('amplitude (a.u.)')
    plt.title("singlet/triplet states ")
    plt.show()