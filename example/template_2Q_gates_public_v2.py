#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 25 Feb

@author: mruss
"""
import numpy as np
import scipy as sp
import pickle
import math

import sys
from datetime import datetime
import  c_solver.DM_solver as DM

import matplotlib.pyplot as plt
from scipy import signal
import qutip as qt
import cmath
import c_solver.pulse_generation.pulse_generic as pgen

from pathos.multiprocessing import ProcessingPool as Pool

global_counter = 0

def exchange_fun(rel_det):
    return np.exp(2.*rel_det)

def exchange_der(rel_det):
    return np.exp(2.*rel_det)*2.

def exchange_inverse(value):
    return np.log(value)/2.




def pulse_auxilary(time, delay, indicator = 0):
    result = 0.
    trapezoid_coefficients = [[0., 0.25,0.75, 1.],
                              [0., 0.2,0.8, 1.]][indicator]
    if time <= trapezoid_coefficients[0]/delay:
        return result
    elif time <= trapezoid_coefficients[1]/delay:
        result = time/trapezoid_coefficients[1]*delay
        return result
    elif time <= trapezoid_coefficients[2]/delay:
        result = 1.
        return result
    elif time <= trapezoid_coefficients[3]/delay:
        result = (1.-(time-trapezoid_coefficients[2]/delay)
                      /trapezoid_coefficients[1]*delay)
        return result
    else:
        return None
    
pulse_auxilary = np.vectorize(pulse_auxilary)
    


def sloped_envelope_rabi(delta_t, sample_rate = 1,indicator = 0):
        """
        function that has window shape at the start and at the end (first 8 and last 8 ns)

        Args:
            delta_t (double) : time in ns of the pulse.
            sample_rate (double) : sampling rate of the pulse (GS/s).

        Returns:
            evelope (np.ndarray) : array of the evelope.
        """
        window = ['boxcar','blackman','blackmanharris',
                  'hann','flattop',('kaiser',14),('tukey',0.5)]
        
        n_points = int(np.ceil(delta_t*sample_rate )-1.)
        envelope = np.empty([n_points], np.double)
        envelope = signal.get_window(window[indicator], n_points*10)[::10]
        compensate_height = n_points/np.sum(envelope)
        envelope = envelope*compensate_height
        return envelope




def sloped_envelope_rabi(delta_t, sample_rate = 1,indicator = 0):
        """
        function that has window shape at the start and at the end (first 8 and last 8 ns)

        Args:
            delta_t (double) : time in ns of the pulse.
            sample_rate (double) : sampling rate of the pulse (GS/s).

        Returns:
            evelope (np.ndarray) : array of the evelope.
        """
        window = ['boxcar','blackman','blackmanharris',
                  'hann','flattop',('kaiser',14),('tukey',0.5)]
        
        n_points = int(np.ceil(delta_t*sample_rate )-1.)
        envelope = np.empty([n_points], np.double)
        envelope = signal.get_window(window[indicator], n_points*10)[::10]
        compensate_height = n_points/np.sum(envelope)
        envelope = envelope*compensate_height
        return envelope
    
def sloped_envelope_rabi_der(delta_t, sample_rate = 1,indicator = 0):
        """
        function that has window shape at the start and at the end (first 8 and last 8 ns)

        Args:
            delta_t (double) : time in ns of the pulse.
            sample_rate (double) : sampling rate of the pulse (GS/s).

        Returns:
            evelope (np.ndarray) : array of the evelope.
        """
        window = ['boxcar','blackman','blackmanharris','hann','flattop',('kaiser',14)]
        
        n_points = int(np.ceil(delta_t*sample_rate )-1.)
        envelope = np.empty([n_points], np.double)
        envelope = signal.get_window(window[indicator], n_points*10)[::10]
        compensate_height = n_points/np.sum(envelope)
        envelope = envelope*compensate_height
        return np.gradient(envelope)*sample_rate
    
def sloped_envelope_rabi_int(delta_t, sample_rate = 1,indicator = 0):
        """
        function that has window shape at the start and at the end (first 8 and last 8 ns)

        Args:
            delta_t (double) : time in ns of the pulse.
            sample_rate (double) : sampling rate of the pulse (GS/s).

        Returns:
            evelope (np.ndarray) : array of the evelope.
        """
        window = ['boxcar','blackman','blackmanharris','hann','flattop',('kaiser',14)]
        
        n_points = int(np.ceil(delta_t*sample_rate )-1.)
        envelope = np.empty([n_points], np.double)
        envelope = signal.get_window(window[indicator], n_points*10)[::10]
        compensate_height = n_points/np.sum(envelope)
        envelope = envelope*compensate_height
        
        integrand = np.zeros([n_points],np.double)
        envelope_sqr = envelope
        for it in range(n_points-1):
            integrand[it+1] = np.sum(envelope_sqr[0:it])*delta_t/n_points
        return integrand

def sloped_envelope_rabi_int_square(delta_t, sample_rate = 1,indicator = 0):
        """
        function that has window shape at the start and at the end (first 8 and last 8 ns)

        Args:
            delta_t (double) : time in ns of the pulse.
            sample_rate (double) : sampling rate of the pulse (GS/s).

        Returns:
            evelope (np.ndarray) : array of the evelope.
        """
        window = ['boxcar','blackman','blackmanharris','hann','flattop',('kaiser',14)]
        
        n_points = int(np.ceil(delta_t*sample_rate )-1.)
        envelope = np.empty([n_points], np.double)
        envelope = signal.get_window(window[indicator], n_points*10)[::10]
        compensate_height = n_points/np.sum(envelope)
        envelope = envelope*compensate_height
        
        integrand = np.zeros([n_points],np.double)
        envelope_sqr = envelope**2
        for it in range(n_points-1):
            integrand[it+1] = np.sum(envelope_sqr[0:it])*delta_t/n_points
        return integrand


###############################################################################
###########################   Two_qubit gates   ###############################
###############################################################################


class two_qubit():
    """docstring for two_qubit_simumlation of cphase gate 
        in rotating frame of global Magnetic field"""
    def __init__(self, 
                 leverarm_exchange = 1.797, 
                 # offset_exchange = np.log(110.2554*1e-6),
                 residual_exchange = 30. ,
                 pulsed_exchange = 8.552 ,
                 b_field_diff_z = 0.110, 
                 delta_f_q0=-0.542*1e-3, 
                 delta_f_q1=4.869*1e-3, 
                 T2star_q0 = 13., 
                 T2star_q1 = 19.,
                 drive_saturation= 5.,
                 drive_asymmetry = 5.,
                 x_gate_time = 300.,
                 waiting_time = 5.
                 ):
        """
        Initialize two-qubit classe
        Args:
            leverarm_exchange (double) : leverarm between barrier gate and exchange (GHz/V)
            residual_exchange (double) : residual exchange interaction (in kHz)
            pulsed_exchange (double) : pulsed exchange interaction (in MHz)
            b_field_diff_z (double) : magnetic field difference (in MHz)
            delta_f_q0 (double) : Change of resonant frequency of qubit 0 from barrier (in GHz/V)
            delta_f_q1 (double) : Change of resonant frequency of qubit 1 from barrier (in GHz/V)
            T2star_q0 (double) : Dephasing time of qubit 1 from Ramsey in (in microseconds)
            T2star_q1 (double) : Dephasing time of qubit 2 from Ramsey in (in microseconds)
            drive_saturation (double) : Saturation of ac drive
            drive_asymmetry (double) : Asymmetry of ac drive (value>1: Q1 drives faster)
            x_gate_time (double) : Gate time of a spin flip
            waiting_time (double) : Waiting time after each gate
        Returns:
            
        """
        
        #Preset fundamental system parameters
        self.delta_z = b_field_diff_z*1e9
        self.vB_leverarm = leverarm_exchange
        self.exchange_residual = 1e3 *residual_exchange
        self.exchange_dc = pulsed_exchange*1e6
        self.offset = np.log(self.exchange_residual*1e-9)
        
        self.delta_freq_q0 = delta_f_q0*1e9
        self.delta_freq_q1 = delta_f_q1*1e9
        
        self.dephasing_Q0 = T2star_q0*1e-6
        self.dephasing_Q1 = T2star_q1*1e-6
        
        self.rabi_pi2_duration = x_gate_time/2.
        self.waiting_time = waiting_time
        
        # set saturation parameters
        self.exchange_sat = self.exchange_residual
        self.drive_sat = drive_saturation
        
        # sets the active value of the exchange interaction to the residual
        
        self.vB_operation_point = exchange_inverse(
            self.exchange_dc/self.exchange_sat)/self.vB_leverarm
        
        #Set up low pass filter frequency
        self.lowpass_filter = 150.*1e6
        
        
        #Preset Hamiltonians
        self.H_zeeman = 2.*np.pi*(qt.tensor(qt.sigmaz(), qt.qeye(2))/4.
                                  -qt.tensor(qt.qeye(2), qt.sigmaz())/4.)[:,:]
        
        self.H_zeeman_Q0 = 2.*np.pi*(qt.tensor(qt.sigmaz(), qt.qeye(2))/2.)[:,:]
        self.H_zeeman_Q1 = 2.*np.pi*(qt.tensor(qt.qeye(2), qt.sigmaz())/2.)[:,:]
        
        self.H_zeeman_zz = 2.*np.pi*(qt.tensor(qt.sigmaz(), qt.sigmaz())/4.)[:,:]
        
        # We start using a symmetric driving Hamiltonian. If necessary this 
        # can generalized to asymmetric drive power and non-linear effects
        self.H_zeeman_ac_raw = 2.*np.pi*(drive_asymmetry
                                     *qt.tensor(qt.sigmax(), qt.qeye(2))/2.
                                     +qt.tensor(qt.qeye(2), qt.sigmax())/2.)[:,:]
        
        self.H_zeeman_ac = self.H_zeeman_ac_raw*self.drive_sat
        
        # Preset exchange Hamiltonian. Multiplication with saturated exchange 
        # interaction is necessary since formula is neither linear nor 
        # exponentional, thus, factors cannot be moved infront of whole formula.
        self.H_heisenberg_raw = 2.*np.pi*((
                qt.tensor(qt.sigmax(), qt.sigmax())
                +qt.tensor(qt.sigmay(), qt.sigmay())
                +qt.tensor(qt.sigmaz(), qt.sigmaz())
                -qt.tensor(qt.qeye(2), qt.qeye(2))
                )/4.)[:,:]
        
        self.H_heisenberg = self.H_heisenberg_raw*self.exchange_sat
        
        
        self.init = np.zeros([4,4], dtype=np.complex)
        self.init[3,3] = 1.
        
        self.readout_fidelities = np.array([0.8,0.77,0.9,0.87],dtype = float)
        
        # define pulse library which is consistent with pyGSTi
        
        self.pulse_library = {
            "Gxpi2:0" : (-1j*np.pi*0.25*qt.tensor(qt.sigmax(), qt.qeye(2))).expm()[:,:],
            "Gypi2:0" : (-1j*np.pi*0.25*qt.tensor(qt.sigmay(), qt.qeye(2))).expm()[:,:],
            "Gxpi2:1" : (-1j*np.pi*0.25*qt.tensor(qt.qeye(2), qt.sigmax())).expm()[:,:],
            "Gypi2:1" : (-1j*np.pi*0.25*qt.tensor(qt.qeye(2), qt.sigmay())).expm()[:,:],
            "Gcphase:0:1" : np.diag(np.array([1.,1.,1.,-1.], dtype = np.complex)),
            "Gii:0:1" : np.diag(np.array([1.,1.,1.,1.], dtype = np.complex))
            }
        
        
        # dummy initialization to prevent errors
        self.phase_library= {}
        
###############################################################################
###########################   Helper functions   ##############################
###############################################################################
        
    def show_pulse_sequence(self,pulse, endtime, sample_rate = 1e11):
        f_sample_rate = sample_rate
        t, v  = pulse.get_pulse(endtime,f_sample_rate)
        plt.plot(t,np.real(v))
        plt.xlabel('time (ns)')
        plt.ylabel('amplitude (a.u.)')
        plt.show()
        
    def show_pulse_sequence_expsat(self,pulse, endtime, sample_rate = 1e11):
        f_sample_rate = sample_rate
        t, v  =pulse.get_pulse(endtime, f_sample_rate)
        v_data = self.exchange_sat*exchange_fun(np.real(v))
        
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(t[0:-2], v_data[0:-2],'b',label = "acswap")
        axs[0].set_xlabel('time (ns)')
        axs[0].set_ylabel('exchange (Hz)')
        axs[0].grid(True)
        
        axs[1].plot(t[0:-2], np.real(v)[0:-2],'b',label = "acswap")
        axs[1].set_xlabel('time (ns)')
        axs[1].set_ylabel('voltage (meV)')
        axs[1].grid(True)

        
        
############################   CPHASE calibration   ###########################
    
    def simulate_cphase_calibration_sequence(self, noise_strength, set_target, 
                         set_flip = None, errors = [0.,0.,0.,0.],
                         exchange = None, pulse_time = None):
        
        [y_error_1, z_error_1, y_error_2, z_error_2] = errors
        
        if exchange is not None:
            self.exchange_dc = exchange*1e9
            self.vB_operation_point = (exchange_inverse(
                (self.exchange_dc-self.exchange_residual)/self.exchange_sat)
                -self.offset)/self.vB_leverarm
        
        if exchange is not None:
            self.exchange_dc = exchange*1e9
            self.vB_operation_point = (exchange_inverse((self.exchange_dc-self.exchange_residual)
                                                            /self.exchange_sat)
                                       )/self.vB_leverarm
        
        print("exchange on: ",self.exchange_dc*1e-6)
        print("vB_operation_point: ",self.vB_operation_point)
        
        
        self.H_static = (self.H_zeeman/(2.*np.pi)*self.delta_z 
                             + self.H_heisenberg_raw/(2.*np.pi) 
                             * self.exchange_residual)
        
        
        self.eig_energies, self.eig_states = sp.linalg.eigh(
            self.H_static+np.diag(1e11*np.array([1.,0.,0.,-1.]))
            )
        
        # feel free to use any input state you like
        self.init = qt.ket2dm(qt.Qobj(self.eig_states[0]))[:,:]
        
        
        self.noise_amplitude_voltage = noise_strength*self.vB_leverarm
        self.pulse_exchange=pgen.pulse()
        
        
        # Define envelope function for exchange pulse
        def envelope_fun(time):
                delay=1.
                pulse_value = pulse_auxilary(time,delay,indicator =1)
                # includes logarithmic compensation
                pulse_value = (
                    (self.exchange_dc-self.exchange_residual)
                    *pulse_value
                    +self.exchange_residual
                    )/self.exchange_sat
                inverse_value = np.log(pulse_value)/2.
                return pulse_value-self.offset*0.
        
        which_peak = 1.
        
        area_pulse, err = sp.integrate.quad(
            lambda f_time: (
                self.exchange_sat*exchange_fun(envelope_fun(f_time))
                -self.exchange_residual),0.,1.)
        
        self.pulse_time = (2.*which_peak-1)*1./2./area_pulse*1e9
        
        
        if pulse_time is not None:
            self.pulse_time = pulse_time
        
        print("Pulse time: ", self.pulse_time)
        
        def add_dc_exchange(self, t_start, t_stop, t_ramp,
                                 f_pulse = self.pulse_exchange):
            """
            adds an ac exchange pulse to the qubit
            Args
                t_ramp (double) : duration of the ramp in ns
                t_start (double) : starting time of the pulse in ns
                t_stop (double) : end time of the pulse in ns
                number_of_ramps_before (int) : number of ramps before (optional)
                f_pulse (pulse) : pulse to which the additional pulse is added (optional)
                f_is_RWA (boolean) : indicator if RWA (rotating wave approximation) should be applied
            Return
                none : 
            """
            
            #create pulse sequence
            f_pulse.add_function(t_start,t_start+t_ramp,envelope_fun)
            f_pulse.add_block(t_start+t_ramp,t_stop,0)
            
        def show_pulse_sequence(pulse):
            t, v  = pulse.get_pulse(self.total_time,1e11)
            plt.plot(t,v)
            plt.xlabel('time (ns)')
            plt.ylabel('amplitude (a.u.)')
            plt.show()
        
        def show_pulse_sequence_exp(pulse):
            f_sample_rate = self.sample_rate*1e9
            t, v  =pulse.get_pulse(self.total_time, f_sample_rate)
            v_data = self.exchange_sat*exchange_fun(np.real(v))
            
            fig, axs = plt.subplots(2, 1)
            axs[0].plot(t[0:-2], np.real(v_data[0:-2]),'b',label = "acswap")
            axs[0].set_xlabel('time (ns)')
            axs[0].set_ylabel('exchange (Hz)')
            axs[0].grid(True)
            
            axs[1].plot(t[0:-2], np.real(v[0:-2])/self.vB_leverarm,'b',label = "acswap")
            axs[1].set_xlabel('time (ns)')
            axs[1].set_ylabel('voltage (meV)')
            axs[1].grid(True)
            plt.show()
            
        
        def get_average_gate_fidelity(self, runs = 1000, target = None, 
                                      sample_rate = 50000, 
                                      charge_noise_amp = 0.):
            """
            returns average gate fidelity
            Args
                runs (int) : number of runs to compute the average gate fidelity
                target (4x4 numpy array) : target unitary to compute fidelity
            """
            oneoverfnoise=lambda omega: 1./2./np.pi/omega
            self.solver_obj.calculate_evolution(
                self.init,self.total_time,
                sample_rate,1)
            U_ideal = self.solver_obj.get_unitary()[0]
            
            U_ideal = ( self.U_rot
                        @ np.conjugate(self.eig_states) 
                        @ U_ideal[:,:]
                        @ np.transpose(self.eig_states)
                        )
            
            number_runs = 1
            
            # skips averaging if noise is expected to be negligible (careful use)
            if charge_noise_amp > 1e-10:
                #set up noise spectrum
                oneoverfnoise=lambda omega: 1./2./np.pi/omega
                number_runs = runs
                self.solver_obj.add_noise_static(self.H_zeeman_Q0,
                                                 self.dephasing_Q0)
                self.solver_obj.add_noise_static(self.H_zeeman_Q1,
                                                 self.dephasing_Q1)
                self.solver_obj.add_noise_generic_exp(self.H_heisenberg,
                                                  oneoverfnoise,
                                                  charge_noise_amp)
            
            self.solver_obj.calculate_evolution(self.init,self.total_time,
                                                sample_rate,int(number_runs))
            
            U_list = self.solver_obj.get_unitary()
            
            
            # Calculate the averaged super operator in the Lioville 
            # superoperator form using column convention
            basis = [qt.basis(4,it) for it in range(4)]
            superoperator_basis = [basis_it1*basis_it2.dag() 
                                   for basis_it2 in basis 
                                   for basis_it1 in basis]
            averaged_map = np.zeros([16,16],dtype=np.complex)
            for u in U_list:
                temp_U = qt.Qobj(( self.U_rot
                                @ np.conjugate(self.eig_states) 
                                @ u[:,:]
                                @ np.transpose(self.eig_states)
                                ))
                # print("unitary Fid: ",np.abs(temp_U)**2.)
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
            if target is None:
                target = sp.linalg.expm(-1.*np.pi*1j*(
                    (qt.tensor(qt.sigmaz(),qt.sigmaz())
                     -qt.tensor(qt.qeye(2),qt.qeye(2)))/4.
                    )[:,:])
            
            
            # get phase from optimizing noiseless unitary evolution
            def to_minimize_fidelity(theta):
                temp_z_gate = np.matmul((sp.linalg.expm(-1j*theta[0]
                                                       *self.H_zeeman_Q0)
                                          @ sp.linalg.expm(-1j*theta[1]
                                                       *self.H_zeeman_Q1)),
                                        U_ideal)
                temp_m = np.matmul(np.conjugate(np.transpose(target)),
                                   temp_z_gate)
                return np.real(1.-(np.trace(
                    np.matmul(temp_m,np.conjugate(np.transpose(temp_m))))
                    +np.abs(np.trace(temp_m))**2.)/20.)
            
            ideal_phase = sp.optimize.minimize(
                to_minimize_fidelity, 
                [0.5,0.5], 
                method='SLSQP',
                bounds=[(0.,2.),(0.,2.)],
                # method='Nelder-Mead', 
                tol=1e-10
                ).x 
            
            target = np.matmul((sp.linalg.expm(1j*ideal_phase[0]*self.H_zeeman_Q0)
                                @ sp.linalg.expm(1j*ideal_phase[1]*self.H_zeeman_Q1)),
                               target)
            
            
            # Change the shape of the averaged super operator to match 
            # the definitions used in QuTip (row convention)
            target = qt.Qobj(target)
            averaged_map = qt.Qobj(averaged_map).trans()
            averaged_map._type = 'super'
            averaged_map.dims = [[[4], [4]], [[4], [4]]]
            averaged_map.superrep  = qt.to_super(target).superrep
            
            fidelity = qt.average_gate_fidelity(averaged_map,target)
            return fidelity
        
        
        self.solver_obj = DM.DM_solver()
        self.solver_obj.add_H0(self.H_zeeman,self.delta_z)
        
        # add negative global voltage offset such that 0 voltage 
        # corresponds to 'offset' value
        self.pulse_exchange.add_offset(0)
        
        
        
        self.start_step =0.
        self.stop_step =0.
        self.stop_step += self.pulse_time+10.
        add_dc_exchange(self,self.start_step,self.stop_step,self.pulse_time)
        
        # set total time and sample rate to 10 ps
        self.total_time = self.stop_step
        self.sample_rate = int(100.*self.total_time)
        
        self.pulse_exchange.add_filter(self.lowpass_filter,False)
        
        # show_pulse_sequence_expsat(self.pulse_exchange)
        self.solver_obj.add_H1(self.H_zeeman_Q0*self.delta_freq_q0,self.pulse_exchange)
        self.solver_obj.add_H1(self.H_zeeman_Q1*self.delta_freq_q1,self.pulse_exchange)
        self.solver_obj.add_H1_exp(self.H_heisenberg,self.pulse_exchange)
        
        self.U_rot = sp.linalg.expm(2.*np.pi*1j*self.total_time*(
                                    np.conjugate(self.eig_states) 
                                    @ self.H_static*1e-9
                                    @ np.transpose(self.eig_states)
                                    )[:,:])
        
        
        global global_counter
        print(global_counter)
        global_counter += 1
        
        self.solver_obj.calculate_evolution(self.init,self.total_time,self.sample_rate,1)
        U = self.solver_obj.get_unitary()[0]
        
        
        U = (np.conjugate(self.eig_states)
             @ U
             @ np.transpose(self.eig_states))
        
        
        U = self.U_rot @ U
        
        angle = np.linspace(0.,2.*np.pi,200)
        prob_0 = np.zeros([len(angle)],float)
        prob_1 = np.zeros([len(angle)],float)
        
        f0 = 0.77
        f1 = 0.81
        
        f0 = 1
        f1 = 1
        
        
        for it in range(len(angle)):
            
            if set_target == 1:
                y_error_target = y_error_1*0.
                z_error_target = z_error_1*0.
                
                paulix = qt.tensor(qt.sigmax(), qt.qeye(2))[:,:]
                pauliy = qt.tensor(qt.sigmay(), qt.qeye(2))[:,:]
                pauliz = qt.tensor(qt.sigmaz(), qt.qeye(2))[:,:]
                pauli0 = qt.qeye(4)[:,:]
                gate_pi = (sp.linalg.expm(-1j*np.pi*qt.tensor(qt.qeye(2), 
                                                             qt.sigmax())[:,:]/2.)
                           @ sp.linalg.expm(-1j*np.pi*(
                           y_error_2*pauliy)/2.))
                povm = np.diag([f1,f1,1-f0,1-f0])
                
            elif set_target == 2:
                y_error_target = y_error_2*0.
                z_error_target = z_error_2*0.
                paulix = qt.tensor(qt.qeye(2), qt.sigmax())[:,:]
                pauliy = qt.tensor(qt.qeye(2), qt.sigmay())[:,:]
                pauliz = qt.tensor(qt.qeye(2), qt.sigmaz())[:,:]
                pauli0 = qt.qeye(4)[:,:]
                gate_pi = (sp.linalg.expm(-1j*np.pi*qt.tensor(qt.sigmax(), 
                                                             qt.qeye(2))[:,:]/2.)
                           @ sp.linalg.expm(-1j*np.pi*(
                           y_error_2*pauliy)/2.))
                povm = np.diag([f1,1-f0,f1,1-f0])
                
            else:
                print("Error: target qubit is not accesable in the current device")
                return None
            
            gate_I =  (sp.linalg.expm(-1j*np.pi*paulix/4.)
                       @ sp.linalg.expm(-1j*np.pi*(
                           y_error_target*pauliy 
                           + z_error_target * pauliz )/4.)
                       )
            gate_M =  (sp.linalg.expm(-1j*np.pi*(np.cos(angle[it])*paulix
                                           +np.sin(angle[it])*pauliy)/4.)
                       @ sp.linalg.expm(-1j*np.pi*(
                           y_error_target*(-np.sin(angle[it])*paulix
                                           +np.cos(angle[it])*pauliy) 
                           + z_error_target * pauliz )/4.)
                       )
            prob_0[it] = np.real(np.trace(
                gate_M @ U 
                @ gate_I @ self.init
                @ gate_I.conj().T @ U.conj().T
                @ gate_M.conj().T @ povm))
            
            prob_1[it] = np.real(np.trace(
                gate_M @ U 
                @ gate_I
                @ gate_pi @ self.init @ gate_pi.conj().T
                @ gate_I.conj().T @ U.conj().T
                @ gate_M.conj().T @ povm))
        
        
        
        fid_normal =get_average_gate_fidelity(
            self,sample_rate = self.sample_rate, 
            charge_noise_amp = self.noise_amplitude_voltage )
        
        print("1-F: ",1-fid_normal)
        
        return angle, prob_0, prob_1
    
    
    
    
###############################################################################
###############################   Gates   #####################################
###############################################################################
    

    
##############################   CPHASE   #####################################
    
    def get_fidelity_cphase(self, noise_strength, b_diff = None, 
                            exchange = None, frequency_debug = None, 
                            amplitude_debug = None, 
                            set_det = None, full_shaping = None,
                            ramp_style = 0, inversion_level = 0):
        
        
        if exchange is not None and set_det is None:
            self.exchange_dc = exchange*1e9
            self.voltage_on = (exchange_inverse(self.exchange_dc
                                                /self.exchange_sat)
                               /self.vB_leverarm)
        elif exchange is None and set_det is not None:
            self.voltage_on = set_det
            self.exchange_dc = self.exchange_sat*exchange_fun(
                self.vB_leverarm*self.voltage_on)
        elif exchange is not None and set_det is not None:
            print("Cannot set exchange and detuning at the same time.")
        
        
        self.noise_amplitude_voltage = noise_strength*self.vB_leverarm
        if b_diff is not None:
            self.delta_z = b_diff*1e9
            
        # set static Hamiltonian for rotating frame transformation
        self.H_static= (self.H_zeeman/(2.*np.pi)*self.delta_z)
        self.eig_energies, self.eig_states = sp.linalg.eigh(
            self.H_static+np.diag(1e10*np.array([1.,0.,0.,-1.]))
            )
        # set dummy initial state (not relevant for fidelity)
        self.init = qt.ket2dm(qt.Qobj(self.eig_states[1]))[:,:]
        
        self.noise_amplitude_voltage = noise_strength*np.abs(self.vB_leverarm)
        
        self.pulse_exchange=pgen.pulse()
        
        # Define envelope function for exchange pulse
        def envelope_fun(time):
                delay=1.
                ramp_value = pulse_auxilary(time,delay,indicator =ramp_style)
                if inversion_level == 1:
                    ramp_value = (
                        (self.exchange_dc-self.exchange_residual)
                        *ramp_value
                        +self.exchange_residual
                        )/self.exchange_sat
                    inverse_value = np.log(ramp_value)/2.
                else:
                    inverse_value = (ramp_value
                                      *self.vB_leverarm
                                      *self.vB_operation_point)
                return inverse_value-self.offset*0.
        
        
        # Defines how many rewinds of cphase are done. Standard is 0 which corresponds 
        # to the first and fastest cphase implementation
        number_rewinds = 0.
        
        area_pulse, err = sp.integrate.quad(
            lambda f_time: (
                self.exchange_sat*exchange_fun(envelope_fun(f_time))
                -self.exchange_residual),0.,1.)
        
        self.pulse_time = (2.*(number_rewinds+1)-1)*1./2./area_pulse*1e9
        
        
        
        
        
        def add_dc_exchange_pulse(self, t_start, t_stop):
            """
            adds an dc exchange pulse to the qubit
            Args
                t_ramp (double) : duration of the ramp in ns
                t_start (double) : starting time of the pulse in ns
                t_stop (double) : end time of the pulse in ns
            Return
                none : 
            """
            
            self.pulse_exchange.add_function(t_start,
                                              t_stop,
                                              envelope_fun)
        
        
        
        def get_unitary_gate_fidelity(self, U = None):
            """
            returns unitary gate fidelity
            Args
                runs (str/tuple) : number of runs to compute the average gate fidelity
            """
            self.solver_obj.calculate_evolution(self.init,self.total_time,
                                                50000,1)
            U = qt.Qobj(self.solver_obj.get_unitary()[0])
    
            target = qt.Qobj(sp.linalg.expm(
                -np.pi/4.*1j*sp.sparse.diags([0.,-1.,-1.,0.]).todense())
                )
            temp_phase= self.delta_z*(self.total_time*1e-9)
            SQphase= qt.Qobj(sp.linalg.expm(-1j*temp_phase*self.H_zeeman))
            fidelity = qt.average_gate_fidelity(U,target*SQphase)
            
            return fidelity
        
        
        
        
        def get_maklin_invariants(self):
            """
            returns maklin invariants of unitary if 2 qubit unitary 
            otherwise yields error
            Args
                runs (str/tuple) : number of runs to compute the 
                average gate fidelity
            """
            self.solver_obj.calculate_evolution(self.init,self.total_time,
                                                self.sample_rate,1)
            U = (qt.Qobj(self.solver_obj.get_unitary()[0]).tidyup())
            
            if np.array_equal(U.dims,[[4],[4]]):
                Q_matrix = np.array([[1,0,0,1j],[0,1j,1,0],
                                     [0,1j,-1,0],[1,0,0,-1j]],
                                    dtype=np.complex)/np.sqrt(2)
                U_matrix = ( self.U_rot
                            @ np.conjugate(self.eig_states) 
                            @ U [:,:]
                            @ np.transpose(self.eig_states)
                            )
                
                m_matrix = ( 
                            (np.transpose(Q_matrix)
                            @ np.transpose(U_matrix) 
                            @ np.conjugate(Q_matrix)
                            @ np.conjugate(np.transpose(Q_matrix))
                            @ U_matrix
                            @ Q_matrix)
                            )
                return [((np.trace(m_matrix))**2
                        *np.linalg.det(np.conjugate(
                            np.transpose(U_matrix)))/16.
                        ),(
                            ((np.trace(m_matrix))**2 
                         - np.trace(m_matrix@m_matrix))
                         *np.linalg.det(np.conjugate(np.transpose(U_matrix)))
                         /4.)]
            else:
                print("Final matrix is not unitary matrix of dimension 4")
                return None , None
        
            
        def get_average_gate_fidelity(self, runs = 1000, target = None, 
                                      sample_rate = 50000, 
                                      charge_noise_amp = 0.):
            """
            returns average gate fidelity
            Args
                runs (int) : number of runs to compute the average gate fidelity
                target (4x4 numpy array) : target unitary to compute fidelity
            """
            oneoverfnoise=lambda omega: 1./2./np.pi/omega
            self.solver_obj.calculate_evolution(
                self.init,self.total_time,
                sample_rate,1)
            U_ideal = self.solver_obj.get_unitary()[0]
            
            U_ideal = ( self.U_rot
                        @ np.conjugate(self.eig_states) 
                        @ U_ideal[:,:]
                        @ np.transpose(self.eig_states)
                        )
            
            number_runs = 1
            
            # skips averaging if noise is expected to be negligible (careful use)
            if charge_noise_amp > 1e-10:
                #set up noise spectrum
                oneoverfnoise=lambda omega: 1./2./np.pi/omega
                number_runs = runs
                self.solver_obj.add_noise_static(self.H_zeeman_Q0,
                                                 self.dephasing_Q0)
                self.solver_obj.add_noise_static(self.H_zeeman_Q1,
                                                 self.dephasing_Q1)
                self.solver_obj.add_noise_generic_exp(self.H_heisenberg,
                                                  oneoverfnoise,
                                                  charge_noise_amp)
            
            self.solver_obj.calculate_evolution(self.init,self.total_time,
                                                sample_rate,int(number_runs))
            
            U_list = self.solver_obj.get_unitary()
            
            # Calculate the averaged super operator in the Lioville 
            # superoperator form using column convention
            basis = [qt.basis(4,it) for it in range(4)]
            superoperator_basis = [basis_it1*basis_it2.dag() 
                                   for basis_it2 in basis 
                                   for basis_it1 in basis]
            averaged_map = np.zeros([16,16],dtype=np.complex)
            for u in U_list:
                temp_U = qt.Qobj(( self.U_rot
                                @ np.conjugate(self.eig_states) 
                                @ u[:,:]
                                @ np.transpose(self.eig_states)
                                ))
                
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
            if target is None:
                target = self.pulse_library["Gcphase:0:1"] 
                
            # get phase from optimizing noiseless unitary evolution
            def to_minimize_fidelity(theta):
                temp_z_gate = np.matmul((sp.linalg.expm(-1j*theta[0]
                                                       *self.H_zeeman_Q0)
                                          @ sp.linalg.expm(-1j*theta[1]
                                                       *self.H_zeeman_Q1)),
                                        U_ideal)
                temp_m = np.matmul(target.T.conj(),
                                   temp_z_gate)
                return np.real(1.-(np.trace(
                    np.matmul(temp_m,np.conjugate(np.transpose(temp_m))))
                    +np.abs(np.trace(temp_m))**2.)/20.)
            
            ideal_phase = sp.optimize.minimize(
                to_minimize_fidelity, 
                [0.5,0.5], 
                method='SLSQP',
                bounds=[(0.,2.),(0.,2.)],
                tol=1e-10
                ).x
            # saves ideal phase into phase library
            self.phase_library["Gcphase:0:1"] = ideal_phase
            
            
            
            target = np.matmul((sp.linalg.expm(1j*ideal_phase[0]*self.H_zeeman_Q0)
                                @ sp.linalg.expm(1j*ideal_phase[1]*self.H_zeeman_Q1)),
                               target)
            
            
            # Change the shape of the averaged super operator to match 
            # the definitions used in QuTip (row convention)
            target = qt.Qobj(target)
            averaged_map = qt.Qobj(averaged_map).trans()
            averaged_map._type = 'super'
            averaged_map.dims = [[[4], [4]], [[4], [4]]]
            averaged_map.superrep  = qt.to_super(target).superrep
            
            fidelity = qt.average_gate_fidelity(averaged_map,target)
            return fidelity
        
        
        def show_pulse_sequence_expsat(pulse):
            f_sample_rate = self.sample_rate*1e9
            t, v  =pulse.get_pulse(self.total_time, f_sample_rate)
            v_data = self.exchange_sat*exchange_fun(np.real(v))
            
            fig, axs = plt.subplots(2, 1)
            axs[0].plot(t[0:-2], v_data[0:-2],'b',label = "acswap")
            axs[0].set_xlabel('time (ns)')
            axs[0].set_ylabel('exchange (Hz)')
            axs[0].grid(True)
            
            axs[1].plot(t[0:-2], np.real(v)[0:-2]/self.vB_leverarm,
                        'b',label = "acswap")
            axs[1].set_xlabel('time (ns)')
            axs[1].set_ylabel('voltage (meV)')
            axs[1].grid(True)
            
            
         # Set up sample rate and total time
        self.total_time = self.pulse_time + 10.
        self.sample_rate = int(100.*self.total_time)
        
        # Set up rotating frame
        self.U_rot = sp.linalg.expm(2.*np.pi*1j*self.total_time*(
                                    np.conjugate(self.eig_states) 
                                    @ self.H_static*1e-9
                                    @ np.transpose(self.eig_states)
                                    )[:,:])
        
        self.solver_obj = DM.DM_solver()
        self.solver_obj.add_H0(self.H_zeeman,self.delta_z)
        
        
        self.pulse_exchange=pgen.pulse()
        self.pulse_exchange.add_filter(self.lowpass_filter,False)
        add_dc_exchange_pulse(self,0.,self.pulse_time)
        self.pulse_exchange.add_offset(0.*self.vB_leverarm)
        
        self.solver_obj.add_H1(self.H_zeeman_Q0*self.delta_freq_q0,self.pulse_exchange)
        self.solver_obj.add_H1(self.H_zeeman_Q1*self.delta_freq_q1,self.pulse_exchange)
        self.solver_obj.add_H1_exp(self.H_heisenberg,self.pulse_exchange)
        # show_pulse_sequence_expsat(self.pulse_exchange)
        
        fid_normal =get_average_gate_fidelity(
            self,sample_rate = self.sample_rate, 
            charge_noise_amp = self.noise_amplitude_voltage )
        
        print("total time: ",self.total_time-10.)
        return fid_normal
    
#############################   resonant drive   ##############################
    
    def get_fidelity_crot(self, noise_strength, set_target = 0, set_phase = 0, 
                          b_diff = None, exchange = None):
        
        self.noise_amplitude_voltage = noise_strength*self.vB_leverarm
        if b_diff is not None:
            self.delta_z = b_diff*1e9
        
        self.H_static= (self.H_zeeman/(2.*np.pi)*self.delta_z 
                        + self.H_heisenberg/(2.*np.pi*self.exchange_sat) 
                        * self.exchange_residual)
        self.eig_energies, self.eig_states = sp.linalg.eigh(
            self.H_static+np.diag(1e10*np.array([1.,0.,0.,-1.]))
            )
        self.init = qt.ket2dm(qt.Qobj(self.eig_states[1]))[:,:]
        
        self.noise_amplitude_voltage = noise_strength*np.abs(self.vB_leverarm)
        
        
        self.rabi_pulse_duration = self.rabi_pi2_duration
        
        
        self.pulse_exchange=pgen.pulse()
        self.pulse_rabi=pgen.pulse()
        
        self.phase_tracker = np.zeros([4])
        
        self.transition_frequencies = np.array(
            [self.eig_energies[1]-self.eig_energies[0]-1e10,
             self.eig_energies[3]-self.eig_energies[1]-1e10,
             self.eig_energies[3]-self.eig_energies[2]-1e10,
             self.eig_energies[2]-self.eig_energies[0]-1e10])
        # print(self.transition_frequencies)
        self.H_zeeman_ac_eff=(np.transpose(self.eig_states)
                              @ self.H_zeeman_ac
                              @ self.eig_states/(np.pi))
        self.rabi_pulse_modifier = np.real(np.array(
            [self.H_zeeman_ac_eff[1,0],
             self.H_zeeman_ac_eff[3,1],
             self.H_zeeman_ac_eff[3,2],
             self.H_zeeman_ac_eff[2,0]]))
        
        def add_ac_pulse(self, t_start, t_stop, which_frequency,
                         f_pulse = self.pulse_rabi, f_is_RWA = True,
                         phase_debugger=0):
            """
            adds an ac rabi pulse to the qubit
            Args
                t_start (double) : starting time of the pulse in ns
                t_stop (double) : end time of the pulse in ns
                which_frequency (int) : set conditional frequency
                f_pulse (pulse) : pulse to which the additional pulse is added (optional)
                f_is_RWA (boolean) : indicator if RWA (rotating wave approximation) should be applied
            Return
                none : 
            """
            shape_index = 6
            #set up  rabi pulse
            rabi_frequency = self.transition_frequencies[which_frequency]
            rabi_amplitude = (1./self.rabi_pulse_modifier[which_frequency]
                              /self.rabi_pulse_duration*1e9)
            # print("rabi frequency: ",rabi_frequency)
            def envelope_rabi_pulse_x(delta_t, sample_rate = 1):
                return sloped_envelope_rabi(
                    delta_t,sample_rate,indicator=shape_index
                    )*rabi_amplitude
            
            
            phase_corr = phase_debugger + self.phase_tracker[which_frequency]
            f_pulse.add_MW_pulse((t_start),(t_stop),1.,rabi_frequency , 
                                 phase = phase_corr - set_phase,
                                 AM = envelope_rabi_pulse_x, 
                                 PM = None, is_RWA = f_is_RWA)
            
        def get_unitary_gate_fidelity(self, U = None):
            """
            returns unitary gate fidelity
            Args
                runs (str/tuple) : number of runs to compute the average gate fidelity
            """
            self.solver_obj.calculate_evolution(self.init,self.total_time,
                                                50000,1)
            U = qt.Qobj(self.solver_obj.get_unitary()[0])
            
            target = qt.Qobj(sp.linalg.expm(
                -np.pi/4.*1j*sp.sparse.diags([0.,-1.,-1.,0.]).todense())
                )
            temp_phase= self.delta_z*(self.total_time*1e-9)
            SQphase= qt.Qobj(sp.linalg.expm(-1j*temp_phase*self.H_zeeman))
            fidelity = qt.average_gate_fidelity(U,target*SQphase)
            
            return fidelity
        
        
        
        
        def get_maklin_invariants(self):
            """
            returns maklin invariants of unitary if 2 qubit unitary 
            otherwise yields error
            Args
                runs (str/tuple) : number of runs to compute the 
                average gate fidelity
            """
            self.solver_obj.calculate_evolution(self.init,self.total_time,
                                                self.sample_rate,1)
            U = (qt.Qobj(self.solver_obj.get_unitary()[0]).tidyup())
            
            if np.array_equal(U.dims,[[4],[4]]):
                Q_matrix = np.array([[1,0,0,1j],[0,1j,1,0],
                                     [0,1j,-1,0],[1,0,0,-1j]],
                                    dtype=np.complex)/np.sqrt(2)
                U_matrix = ( self.U_rot
                            @ np.conjugate(self.eig_states) 
                            @ U [:,:]
                            @ np.transpose(self.eig_states)
                            )
                # print("Unitary matrix 2: ", U_matrix[1:3,1:3])
                m_matrix = ( 
                            (np.transpose(Q_matrix)
                            @ np.transpose(U_matrix) 
                            @ np.conjugate(Q_matrix)
                            @ np.conjugate(np.transpose(Q_matrix))
                            @ U_matrix
                            @ Q_matrix)
                            )
                return [((np.trace(m_matrix))**2
                        *np.linalg.det(np.conjugate(
                            np.transpose(U_matrix)))/16.
                        ),(
                            ((np.trace(m_matrix))**2 
                         - np.trace(m_matrix@m_matrix))
                         *np.linalg.det(np.conjugate(np.transpose(U_matrix)))
                         /4.)]
            else:
                print("Final matrix is not unitary matrix of dimension 4")
                return None , None
        
        
        def get_average_gate_fidelity(self, runs = 1000, target_freq = set_target, 
                                      sample_rate = 50000, 
                                      charge_noise_amp = 0.):
            """
            returns average gate fidelity
            Args
                runs (int) : number of runs to compute the average gate fidelity
                target (4x4 numpy array) : target unitary to compute fidelity
            """
            oneoverfnoise=lambda omega: 1./2./np.pi/omega
            self.solver_obj.calculate_evolution(
                self.init,self.total_time,
                sample_rate,1)
            U_ideal = self.solver_obj.get_unitary()[0]
            
            U_ideal = ( self.U_rot
                        @ np.conjugate(self.eig_states) 
                        @ U_ideal[:,:]
                        @ np.transpose(self.eig_states)
                        )

            
            number_runs = 1
            # skips averaging if noise is expected to be negligible (careful use)
            if charge_noise_amp > 1e-10:
                #set up noise spectrum
                oneoverfnoise=lambda omega: 1./2./np.pi/omega
                number_runs = runs
                self.solver_obj.add_noise_static(self.H_zeeman_Q0,
                                                 self.dephasing_Q0)
                self.solver_obj.add_noise_static(self.H_zeeman_Q1,
                                                 self.dephasing_Q1)
                self.solver_obj.add_noise_generic_exp(self.H_heisenberg,
                                                  oneoverfnoise,
                                                  charge_noise_amp)
            
            self.solver_obj.calculate_evolution(self.init,self.total_time,
                                                sample_rate,int(number_runs))
            
            U_list = self.solver_obj.get_unitary()
            
            
            # Calculate the averaged super operator in the Lioville 
            # superoperator form using column convention
            basis = [qt.basis(4,it) for it in range(4)]
            superoperator_basis = [basis_it1*basis_it2.dag() 
                                   for basis_it2 in basis 
                                   for basis_it1 in basis]
            averaged_map = np.zeros([16,16],dtype=np.complex)
            for u in U_list:
                temp_U = qt.Qobj(( self.U_rot
                                @ np.conjugate(self.eig_states) 
                                @ u[:,:]
                                @ np.transpose(self.eig_states)
                                ))
                # print("unitary Fid: ",np.abs(temp_U)**2.)
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
            
            t_elem_1 = [1,3,3,2][target_freq]
            t_elem_2 = [0,1,2,0][target_freq]
            t_matrix = np.zeros([4,4],dtype = np.complex)
            t_matrix[t_elem_1,t_elem_2] = 0.5 * np.exp(-1j*set_phase)
            t_matrix[t_elem_2,t_elem_1] = 0.5 * np.exp(1j*set_phase)
            target = sp.linalg.expm(-1.*np.pi*1j*t_matrix)
            
            transition = [[0.,0.,1.,-1.],[-1.,0.,1.,0.],
                           [-1.,1.,0.,0.],[0.,1.,0.,-1.]][target_freq]
            
            corr_matrix = np.diag(transition)
            
            # get phase from optimizing noiseless unitary evolution
            def to_minimize_fidelity(theta):
                # temp_z_gate = np.matmul(sp.linalg.expm(-1j*theta
                #                                        *self.H_zeeman_Q0),
                #                         U_ideal)
                temp_z_gate = np.matmul(sp.linalg.expm(-1j*theta
                                                       *corr_matrix),
                                        U_ideal)
                temp_m = np.matmul(np.conjugate(np.transpose(target)),
                                   temp_z_gate)
                return np.real(1.-(np.trace(
                    np.matmul(temp_m,np.conjugate(np.transpose(temp_m))))
                    +np.abs(np.trace(temp_m))**2.)/20.)
            
            ideal_phase = sp.optimize.minimize(
                to_minimize_fidelity, 
                [0.5], 
                method='Nelder-Mead', tol=1e-10
                ).x[0]
            
            self.phase_correction_crot = [ideal_phase[0],ideal_phase[1]]
            target = np.matmul(sp.linalg.expm(1j*ideal_phase*corr_matrix),
                               target)
            
            # Change the shape of the averaged super operator to match 
            # the definitions used in QuTip (row convention)
            target = qt.Qobj(target)
            averaged_map = qt.Qobj(averaged_map).trans()
            averaged_map._type = 'super'
            averaged_map.dims = [[[4], [4]], [[4], [4]]]
            averaged_map.superrep  = qt.to_super(target).superrep
            
            fidelity = qt.average_gate_fidelity(averaged_map,target)
            return fidelity
        
        # Set up sample rate and total time
        self.total_time = self.rabi_pulse_duration + 10.
        self.sample_rate = int(100.*self.total_time)
        
        # Set up rotating frame
        self.U_rot = sp.linalg.expm(2.*np.pi*1j*self.total_time*(
                                    np.conjugate(self.eig_states) 
                                    @ self.H_static*1e-9
                                    @ np.transpose(self.eig_states)
                                    )[:,:])
        
        self.solver_obj = DM.DM_solver()
        self.solver_obj.add_H0(self.H_zeeman,self.delta_z)
        
        
        self.pulse_exchange=pgen.pulse()
        self.pulse_exchange.add_filter(self.lowpass_filter,False)
        add_ac_pulse(self,0.,self.rabi_pulse_duration,
                     set_target, f_is_RWA = True)
        self.pulse_exchange.add_offset(self.voltage_off*self.vB_leverarm)
        self.pulse_rabi.add_filter(self.lowpass_filter,False)
        self.solver_obj.add_H1_exp(self.H_heisenberg,self.pulse_exchange)
        self.solver_obj.add_H1_RWA(self.H_zeeman_ac,self.pulse_rabi)
        
        global global_counter
        print(global_counter)
        global_counter += 1
        
        fid_normal =get_average_gate_fidelity(
            self,sample_rate = self.sample_rate, 
            charge_noise_amp = self.noise_amplitude_voltage)
        
        print("total time: ",self.total_time-10.)
        return fid_normal
    
    
#############################   resonant drive   ##############################
    
    
    def get_fidelity_rot(self, noise_strength, set_qubit = 0, set_phase = 0, 
                          b_diff = None, exchange = None):
        
        self.noise_amplitude_voltage = noise_strength*self.vB_leverarm
        if b_diff is not None:
            self.delta_z = b_diff*1e9
        
        self.H_static= self.H_zeeman/(2.*np.pi)*self.delta_z
        self.eig_energies, self.eig_states = sp.linalg.eigh(
            self.H_static+np.diag(1e10*np.array([1.,0.,0.,-1.]))
            )
        self.init = qt.ket2dm(qt.Qobj(self.eig_states[1]))[:,:]
        self.noise_amplitude_voltage = noise_strength*np.abs(self.vB_leverarm)
        
        
        self.rabi_pulse_duration = self.rabi_pi2_duration
        
        
        self.pulse_exchange=pgen.pulse()
        self.pulse_rabi=pgen.pulse()
        
        self.phase_tracker = np.zeros([2])
        
        self.transition_frequencies = np.array(
            [self.eig_energies[2]-self.eig_energies[0]-1e10,
             self.eig_energies[1]-self.eig_energies[0]-1e10])
        # print(self.transition_frequencies)
        self.H_zeeman_ac_eff=(np.transpose(self.eig_states)
                              @ self.H_zeeman_ac
                              @ self.eig_states/(np.pi))
        self.rabi_pulse_modifier = np.real(np.array(
            [self.H_zeeman_ac_eff[2,0],
             self.H_zeeman_ac_eff[1,0]]))
        
        def add_ac_pulse(self, t_start, t_stop, which_frequency,
                         f_pulse = self.pulse_rabi, f_is_RWA = True,
                         phase_debugger=0):
            """
            adds an ac rabi pulse to the qubit
            Args
                t_start (double) : starting time of the pulse in ns
                t_stop (double) : end time of the pulse in ns
                which_frequency (int) : set conditional frequency
                f_pulse (pulse) : pulse to which the additional pulse is added (optional)
                f_is_RWA (boolean) : indicator if RWA (rotating wave approximation) should be applied
            Return
                none : 
            """
            shape_index = 6
            #set up  rabi pulse
            rabi_frequency = self.transition_frequencies[which_frequency]
            rabi_amplitude = (1./self.rabi_pulse_modifier[which_frequency]
                              /self.rabi_pulse_duration*1e9)/2.
            # print("rabi frequency: ",rabi_frequency)
            def envelope_rabi_pulse_x(delta_t, sample_rate = 1):
                return sloped_envelope_rabi(
                    delta_t,sample_rate,indicator=shape_index
                    )*rabi_amplitude
            
            
            phase_corr = phase_debugger + self.phase_tracker[which_frequency]
            f_pulse.add_MW_pulse((t_start),(t_stop),1.,rabi_frequency , 
                                 phase = phase_corr - set_phase,
                                 AM = envelope_rabi_pulse_x, 
                                 PM = None, is_RWA = f_is_RWA)
            
        
        
        def get_maklin_invariants(self):
            """
            returns maklin invariants of unitary if 2 qubit unitary 
            otherwise yields error
            Args
                runs (str/tuple) : number of runs to compute the 
                average gate fidelity
            """
            self.solver_obj.calculate_evolution(self.init,self.total_time,
                                                self.sample_rate,1)
            U = (qt.Qobj(self.solver_obj.get_unitary()[0]).tidyup())
            
            if np.array_equal(U.dims,[[4],[4]]):
                Q_matrix = np.array([[1,0,0,1j],[0,1j,1,0],
                                     [0,1j,-1,0],[1,0,0,-1j]],
                                    dtype=np.complex)/np.sqrt(2)
                U_matrix = ( self.U_rot
                            @ np.conjugate(self.eig_states) 
                            @ U [:,:]
                            @ np.transpose(self.eig_states)
                            )
                # print("Unitary matrix 2: ", U_matrix[1:3,1:3])
                m_matrix = ( 
                            (np.transpose(Q_matrix)
                            @ np.transpose(U_matrix) 
                            @ np.conjugate(Q_matrix)
                            @ np.conjugate(np.transpose(Q_matrix))
                            @ U_matrix
                            @ Q_matrix)
                            )
                return [((np.trace(m_matrix))**2
                        *np.linalg.det(np.conjugate(
                            np.transpose(U_matrix)))/16.
                        ),(
                            ((np.trace(m_matrix))**2 
                         - np.trace(m_matrix@m_matrix))
                         *np.linalg.det(np.conjugate(np.transpose(U_matrix)))
                         /4.)]
            else:
                print("Final matrix is not unitary matrix of dimension 4")
                return None , None
        
        
    
        def get_average_gate_fidelity(self, runs = 1000, 
                                      sample_rate = 50000, 
                                      charge_noise_amp = 0.):
            """
            returns average gate fidelity
            Args
                runs (int) : number of runs to compute the average gate fidelity
                target (4x4 numpy array) : target unitary to compute fidelity
            """
            oneoverfnoise=lambda omega: 1./2./np.pi/omega
            self.solver_obj.calculate_evolution(
                self.init,self.total_time,
                sample_rate,1)
            U_ideal = self.solver_obj.get_unitary()[0]
            
            U_ideal = ( self.U_rot
                        @ np.conjugate(self.eig_states) 
                        @ U_ideal[:,:]
                        @ np.transpose(self.eig_states)
                        )
            number_runs = 1
            
            # skips averaging if noise is expected to be negligible (careful use)
            if charge_noise_amp > 1e-10:
                #set up noise spectrum
                oneoverfnoise=lambda omega: 1./2./np.pi/omega
                number_runs = runs
                self.solver_obj.add_noise_static(self.H_zeeman_Q0,
                                                 self.dephasing_Q0)
                self.solver_obj.add_noise_static(self.H_zeeman_Q1,
                                                 self.dephasing_Q1)
                if self.set_SAT == True:
                    self.solver_obj.add_noise_generic_expsat(self.H_heisenberg,
                                                      oneoverfnoise,
                                                      charge_noise_amp)
                else:
                    self.solver_obj.add_noise_generic_exp(self.H_heisenberg,
                                                  oneoverfnoise,
                                                  charge_noise_amp)
            
            self.solver_obj.calculate_evolution(self.init,self.total_time,
                                                sample_rate,int(number_runs))
            
            U_list = self.solver_obj.get_unitary()
            
            # Calculate the averaged super operator in the Lioville 
            # superoperator form using column convention
            basis = [qt.basis(4,it) for it in range(4)]
            superoperator_basis = [basis_it1*basis_it2.dag() 
                                   for basis_it2 in basis 
                                   for basis_it1 in basis]
            averaged_map = np.zeros([16,16],dtype=np.complex)
            for u in U_list:
                temp_U = qt.Qobj(( self.U_rot
                                @ np.conjugate(self.eig_states) 
                                @ u[:,:]
                                @ np.transpose(self.eig_states)
                                ))
                
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
            
            target  = self.pulse_library["Gxpi2:"+str(set_qubit)]
            if math.isclose(set_phase,0.):
                target  = self.pulse_library["Gxpi2:"+str(set_qubit)]
            elif math.isclose(set_phase,np.pi/2.):
                target  = self.pulse_library["Gypi2:"+str(set_qubit)]
            else:
                print("arbitrary angles not implemented yet.")
            
            
            # get phase from optimizing noiseless unitary evolution
            def to_minimize_fidelity(theta):
                temp_z_gate = np.matmul((sp.linalg.expm(-1j*theta[0]
                                                       *self.H_zeeman_Q0)
                                          @ sp.linalg.expm(-1j*theta[1]
                                                       *self.H_zeeman_Q1)),
                                        U_ideal)
                temp_m = np.matmul(target.T.conj(),
                                   temp_z_gate)
                return np.real(1.-(np.trace(
                    np.matmul(temp_m,np.conjugate(np.transpose(temp_m))))
                    +np.abs(np.trace(temp_m))**2.)/20.)
            
            ideal_phase = sp.optimize.minimize(
                to_minimize_fidelity, 
                [1.,1.], 
                method='SLSQP',
                bounds=[(0.,2.),(0.,2.)],
                tol=1e-10
                ).x
            
            
            self.phase_library["Gx"+"pi2:"+str(set_qubit)] = ideal_phase
            
            target = np.matmul((sp.linalg.expm(1j*ideal_phase[0]*self.H_zeeman_Q0)
                                @ sp.linalg.expm(1j*ideal_phase[1]*self.H_zeeman_Q1)),
                               target)
            
            
            # Change the shape of the averaged super operator to match 
            # the definitions used in QuTip (row convention)
            target = qt.Qobj(target)
            averaged_map = qt.Qobj(averaged_map).trans()
            averaged_map._type = 'super'
            averaged_map.dims = [[[4], [4]], [[4], [4]]]
            averaged_map.superrep  = qt.to_super(target).superrep
            
            fidelity = qt.average_gate_fidelity(averaged_map,target)
            return fidelity
        
        # Set up sample rate and total time
        self.total_time = self.rabi_pulse_duration + self.waiting_time
        self.sample_rate = int(100.*self.total_time)
        # Set up rotating frame
        self.U_rot = sp.linalg.expm(2.*np.pi*1j*self.total_time*(
                                    np.conjugate(self.eig_states) 
                                    @ self.H_static*1e-9
                                    @ np.transpose(self.eig_states)
                                    )[:,:])
        
        self.solver_obj = DM.DM_solver()
        self.solver_obj.add_H0(self.H_zeeman,self.delta_z)
        self.solver_obj.add_H0(self.H_zeeman_LF,self.LO_freq )
        
        
        self.pulse_exchange=pgen.pulse()
        self.pulse_exchange.add_filter(self.lowpass_filter,False)
        add_ac_pulse(self,0.,self.rabi_pulse_duration, set_qubit)
        self.pulse_exchange.add_offset(self.offset)
        self.pulse_rabi.add_offset(self.drive_offset)
        self.pulse_rabi.add_filter(self.lowpass_filter,False)
        
        self.solver_obj.add_H1_exp(self.H_heisenberg,self.pulse_exchange)
        self.solver_obj.add_H1_RWA(self.H_zeeman_ac_raw,self.pulse_rabi)
        

        pulse_exchange_temp = pgen.pulse()
        pulse_exchange_temp = pulse_exchange_temp.__add__(self.pulse_exchange)
        pulse_exchange_temp.add_block(0.,self.total_time,-self.offset)
        
        
        self.solver_obj.add_H1(self.H_zeeman_Q0*self.delta_freq_q0,pulse_exchange_temp)
        self.solver_obj.add_H1(self.H_zeeman_Q1*self.delta_freq_q1,pulse_exchange_temp)
        
        
        global global_counter
        print(global_counter)
        global_counter += 1
        
        fid_normal =get_average_gate_fidelity(
            self,sample_rate = self.sample_rate, 
            charge_noise_amp = self.noise_amplitude_voltage)
        
        
        print("total time: ",self.total_time)
        return fid_normal
    
#########################   compute gate sequence   ###########################
    
    def initialize_gate_sequence(self, set_RWA = True, init_phase_library = True):
        self.total_time_sequence = 0.
        self.pulse_exchange_sequence=pgen.pulse()
        self.pulse_rabi_sequence=pgen.pulse()
        self.pulse_shift_q1=pgen.pulse()
        self.pulse_shift_q2=pgen.pulse()
        self.pulse_exchange_sequence.add_offset(0.*self.vB_leverarm)
        
        self.set_RWA = set_RWA
        
        self.H_static_sequence= self.H_zeeman/(2.*np.pi)*self.delta_z 
        self.eig_energies_sequence, self.eig_states_sequence = sp.linalg.eigh(
            self.H_static_sequence+np.diag(1e10*np.array([1.,0.,0.,-1.]))
            )
        
        self.H_static_basis= (self.H_zeeman/(2.*np.pi)*self.delta_z 
                        + self.H_heisenberg/(2.*np.pi*self.exchange_sat) 
                        * self.exchange_residual)
        self.eig_energies_basis, self.eig_states_basis = sp.linalg.eigh(
            self.H_static_basis+np.diag(1e10*np.array([1.,0.,0.,-1.]))
            )
        
        
        self.transition_frequencies = np.array(
            [self.eig_energies_sequence[2]-self.eig_energies_sequence[0]-1e10,
             self.eig_energies_sequence[1]-self.eig_energies_sequence[0]-1e10])
        # print(self.transition_frequencies)
        self.H_zeeman_ac_eff=(np.transpose(self.eig_states_sequence)
                              @ self.H_zeeman_ac
                              @ self.eig_states_sequence/(np.pi))
        self.rabi_pulse_modifier = np.real(np.array(
            [self.H_zeeman_ac_eff[2,0],
             self.H_zeeman_ac_eff[1,0]]))
        
        
        self.init_sequence = qt.ket2dm(qt.Qobj(self.eig_states_sequence[3]))[:,:]
        self.charge_noise = 0.
        self.phase_tracker_sequence = np.zeros([2],dtype=float)
        self.gate_sequence= []
        if init_phase_library == True:
            self.phase_library.update({'Gcphase:0:1' : np.array([0.63716059,0.84332183])*0.,
                                        'Gxpi2:0' : np.array([0.99994777,0.99768206])*0., 
                                        'Gxpi2:1': np.array([1.00231793,1.0000522])*0.})
        self.solver_obj_sequence = DM.DM_solver()

    def calibrate_phase_acquisition(self):
        
        print("cphase fidelity is: ",self.get_fidelity_cphase(0.))
        print("qubit 0 fidelity is: ",self.get_fidelity_rot(0.,set_qubit=0,
                                                                  set_phase=0.))
        print("qubit 1 fidelity is: ",self.get_fidelity_rot(0.,set_qubit=1,
                                                                  set_phase=0.))
        
        print("phase library: ",self.phase_library['Gcphase:0:1'],
              self.phase_library['Gxpi2:0'],
              self.phase_library['Gxpi2:1'])
        
        return None
    
    def add_cphase(self,full_shaping = None,
                   ramp_style = 0, 
                   inversion_level = 0):
        
        # Define envelope function for exchange pulse
        def envelope_fun(time):
                delay=1.
                ramp_value = pulse_auxilary(time,delay,indicator =ramp_style)
                if inversion_level == 1:
                    ramp_value = (
                        (self.exchange_dc-self.exchange_residual)
                        *ramp_value
                        +self.exchange_residual
                        )/self.exchange_sat
                    inverse_value = np.log(ramp_value)/2.
                else:
                    inverse_value = (ramp_value
                                      *self.vB_leverarm
                                      *self.vB_operation_point)
                    # print(ramp_value)
                return inverse_value-self.offset*0.
        
        
        # Defines how many rewinds of cphase are done. Standard is 0 which corresponds 
        # to the first and fastest cphase implementation
        number_rewinds = 0.
        
        area_pulse, err = sp.integrate.quad(
            lambda f_time: (
                self.exchange_sat*exchange_fun(envelope_fun(f_time))
                -self.exchange_residual),0.,1.)
        
        self.pulse_time = (2.*(number_rewinds+1)-1)*1./2./area_pulse*1e9
        
        self.pulse_exchange_sequence.add_function(self.total_time_sequence,
                                                  self.total_time_sequence 
                                                  + self.pulse_time,
                                              envelope_fun)
        self.total_time_sequence += self.pulse_time + 10.
        
        self.phase_tracker_sequence += np.array(self.phase_library["Gcphase:0:1"])
        
        self.gate_sequence.append(self.pulse_library["Gcphase:0:1"])
        
        return None
    
    def add_rot(self, set_target = 0, set_phase = "x"):
        
        get_phase = {"x" : 0.,
                     "y" : np.pi/2}
        def add_ac_pulse(self, t_start, t_stop, which_frequency,
                         f_pulse = self.pulse_rabi_sequence, f_is_RWA = True):
            """
            adds an ac rabi pulse to the qubit
            Args
                t_start (double) : starting time of the pulse in ns
                t_stop (double) : end time of the pulse in ns
                which_frequency (int) : set conditional frequency
                f_pulse (pulse) : pulse to which the additional pulse is added (optional)
                f_is_RWA (boolean) : indicator if RWA (rotating wave approximation) should be applied
            Return
                none : 
            """
            shape_index = 6
            #set up rabi pulse
            rabi_frequency = self.transition_frequencies[which_frequency]
            rabi_amplitude = (1./self.rabi_pulse_modifier[which_frequency]
                              /self.rabi_pi2_duration*1e9)/2.
            
            def envelope_rabi_pulse_x(delta_t, sample_rate = 1):
                return sloped_envelope_rabi(
                    delta_t,sample_rate,indicator=shape_index
                    )*rabi_amplitude
            
            
            phase_corr = (self.phase_tracker_sequence[which_frequency])*2.*np.pi
            
            f_pulse.add_MW_pulse((t_start),(t_stop),1.,rabi_frequency , 
                                 phase = phase_corr - get_phase[set_phase],
                                 AM = envelope_rabi_pulse_x, 
                                 PM = None, is_RWA = f_is_RWA)
        
        add_ac_pulse(self,self.total_time_sequence,
                     self.total_time_sequence + self.rabi_pi2_duration,
                     set_target, f_is_RWA = self.set_RWA)
        
        self.total_time_sequence += self.rabi_pi2_duration + self.waiting_time
        
        self.phase_tracker_sequence += self.phase_library[
            "Gx"+"pi2:"+str(set_target)]
        
        self.gate_sequence.append(self.pulse_library["G"+ set_phase 
                                                    +"pi2:"
                                                    +str(set_target)])
        
        
        return None
    
    def add_idle(self,waiting_time):
        self.total_time_sequence += waiting_time
    
    
    def compute_gate_sequence(self, noise_strength, runs = 100):
        
        charge_noise_amp = noise_strength
        
        print("total time: ", self.total_time_sequence)
        
        self.sample_rate = int(100.*self.total_time_sequence)
        
        # Set up rotating frame
        U_rot_sequence = sp.linalg.expm(2.*np.pi*1j*self.total_time_sequence*(
                                np.conjugate(self.eig_states_sequence) 
                                @ self.H_static_sequence*1e-9
                                @ np.transpose(self.eig_states_sequence)
                                )[:,:])
        
        
        self.solver_obj_sequence.add_H0(self.H_zeeman,self.delta_z)
        
        self.pulse_exchange_sequence.add_block(0.,self.total_time_sequence,0.)
        self.pulse_rabi_sequence.add_block(0.,self.total_time_sequence,0.)
        
        self.pulse_exchange_sequence.add_filter(self.lowpass_filter,False)
        self.pulse_rabi_sequence.add_filter(self.lowpass_filter,False)
        self.solver_obj_sequence.add_H1_exp(self.H_heisenberg,
                                            self.pulse_exchange_sequence)
        self.solver_obj_sequence.add_H1(self.H_zeeman_Q0*self.delta_freq_q0,
                                        self.pulse_exchange_sequence)
        self.solver_obj_sequence.add_H1(self.H_zeeman_Q1*self.delta_freq_q1,
                                        self.pulse_exchange_sequence)
        if self.set_RWA == True:
            self.solver_obj_sequence.add_H1_RWA(self.H_zeeman_ac,
                                                self.pulse_rabi_sequence)
        else:
            self.solver_obj_sequence.add_H1(self.H_zeeman_ac,
                                            self.pulse_rabi_sequence)
        
        
        oneoverfnoise=lambda omega: 1./2./np.pi/omega
        
        
        number_runs = 1
        
        # skips averaging if noise is expected to be negligible (careful use)
        if charge_noise_amp > 1e-10:
            #set up noise spectrum
            oneoverfnoise=lambda omega: 1./2./np.pi/omega
            number_runs = runs
            self.solver_obj_sequence.add_noise_static(self.H_zeeman_Q0,
                                             self.dephasing_Q0)
            self.solver_obj_sequence.add_noise_static(self.H_zeeman_Q1,
                                             self.dephasing_Q1)
            self.solver_obj_sequence.add_noise_generic_exp(self.H_heisenberg,
                                              oneoverfnoise,
                                              charge_noise_amp)
        
        self.solver_obj_sequence.calculate_evolution(self.init,self.total_time_sequence,
                                            self.sample_rate,int(number_runs))
        
        U_list = self.solver_obj_sequence.get_unitary()
        
        self.final_DM_sequence = (np.conjugate(self.eig_states_sequence) 
                                  @ self.solver_obj_sequence.get_last_density_matrix()
                                  @ np.transpose(self.eig_states_sequence))
        
        # Calculate the averaged super operator in the Lioville 
        # superoperator form using column convention
        basis = [qt.basis(4,it) for it in range(4)]
        superoperator_basis = [basis_it1*basis_it2.dag() 
                               for basis_it2 in basis 
                               for basis_it1 in basis]
        averaged_map = np.zeros([16,16],dtype=np.complex)
        for u in U_list:
            temp_U = (U_rot_sequence
                        @ np.conjugate(self.eig_states_sequence) 
                        @ u[:,:]
                        @ np.transpose(self.eig_states_sequence)
                        )
            temp_U = (sp.linalg.expm(-1j*self.phase_tracker_sequence[0]
                                     *self.H_zeeman_Q0)
                      @ sp.linalg.expm(-1j*self.phase_tracker_sequence[1]
                                       *self.H_zeeman_Q1)
                      @ temp_U)
            
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
        
        
        target = np.identity(4, dtype=np.complex)
        for unitary in self.gate_sequence:
            target = unitary @ target
        
        # Change the shape of the averaged super operator to match 
        # the definitions used in QuTip (row convention)
        target = qt.Qobj(target)
        
        
        
        averaged_map = qt.Qobj(averaged_map).trans()
        averaged_map._type = 'super'
        averaged_map.dims = [[[4], [4]], [[4], [4]]]
        averaged_map.superrep  = qt.to_super(target).superrep
        
        self.target_sequence = target
        self.averaged_map_sequence = averaged_map
        
        
    
    def return_computed_fidelity(self):
        fidelity = qt.average_gate_fidelity(self.averaged_map_sequence,
                                            self.target_sequence)
        
        
        return fidelity
    
    def return_measurement_probabilities(self,total_count = 10000):
        
        list_povm = []
        for it1 in range(2):
            for it2 in range(2):
                povm1 = np.diag([(it1+(-1)**it1*self.readout_fidelities[0]),
                                 ((1-it1)+(-1)**(1-it1)*self.readout_fidelities[1])])
                povm2 = np.diag([(it2+(-1)**it2*self.readout_fidelities[2]),
                                 ((1-it2)+(-1)**(1-it2)*self.readout_fidelities[3])])
                povm=np.kron(povm1,povm2)
                list_povm.append(np.trace(povm @ self.final_DM_sequence))
        
        counts = np.round(np.real(np.array(list_povm))*total_count).astype(int).tolist()
        # print(counts)
        
        return counts
    
    
    
    def compute_mappings(self, plot = False):
        self.dim = 4
        
        temp_basis = [qt.qeye(2),qt.sigmax(),qt.sigmay(),qt.sigmaz()]
        self.pauli_basis = [qt.tensor(iter_1,iter_2)
                            for iter_1 in temp_basis
                            for iter_2 in temp_basis]
        basic_basis = ["I","X","Y","Z"]
        op_label = []
        for iterator1 in range(4):
            for iterator2 in range(4):
                temp_string = basic_basis[iterator1]+basic_basis[iterator2]
                op_label.append(temp_string)
        
        
        choi = self.averaged_map_sequence.dag()
        chi = qt.qpt(choi,[self.pauli_basis])
        ptm = np.zeros([self.dim**2,self.dim**2],dtype = np.complex)
        for it_i in range(self.dim**2):
            pauli_i = self.pauli_basis[it_i]
            for it_j in range(self.dim**2):
                pauli_j = self.pauli_basis[it_j]
                for it_k in range(self.dim**2):
                    pauli_k = self.pauli_basis[it_k]
                    for it_l in range(self.dim**2):
                        pauli_l = self.pauli_basis[it_l]
                        ptm[it_i,it_j] += 1./self.dim*chi[:,:][it_k,it_l]*(
                            np.trace(pauli_i[:,:] @ pauli_k[:,:]
                                     @ pauli_j[:,:] @ pauli_l[:,:]))
        
        # plt.show()
        if plot == True:
            qt.qpt_plot_combined(ptm, [op_label],title = "ptm")
            plt.show()
        
        return ptm
    
    def plot_gate_sequence(self):
        self.show_pulse_sequence(self.pulse_rabi_sequence,
                                 self.total_time_sequence)
        self.show_pulse_sequence_expsat(self.pulse_exchange_sequence,
                                  self.total_time_sequence)
        
        return None


###############################################################################
###############################################################################
######################### end of two qubit gates function #####################
######################### start of auxilary function ##########################
###############################################################################
###############################################################################


if __name__ == '__main__':
    gate = two_qubit()
    gate.initialize_gate_sequence()
    # gate.calibrate_phase_acquisition()
    
    # # add cphase
    # gate.add_rot(0,"y")
    
    # compute sequence
    # gate.compute_gate_sequence(0, runs=1000)
    
    # gate.plot_gate_sequence()
    # print(np.around(gate.compute_mappings(),4))
    # add single qubit rotation gate
    # gate.add_rot(0,"y")
    
    import time
    time_start = time.time()
    list_noise = []
    list_prob = []
    for std_volt in np.logspace(-5,-2,51):
        noise= (std_volt*gate.vB_leverarm)**2.
        gate.initialize_gate_sequence(init_phase_library=False)
        gate.add_cphase()
        gate.compute_gate_sequence(noise, runs=1000)
        list_noise.append([noise,gate.return_computed_fidelity()])
        # print("Noise is: ", noise, " The computed fidelity is: ", gate.return_computed_fidelity())
    
    # noise_threshold = 0.000469537
    time_stop = time.time()
    # np.savetxt("./noise_vs_CPfidelity.csv", np.array(list_prob), delimiter=',')
    print(" Computation time is: ", (time_stop-time_start)/3600., " hours")
    
    plt.plot(noise[:,0],noise[:,1])
    plt.xlabel('noise (ns)')
    plt.ylabel('amplitude (a.u.)')
    plt.show()
    
    
    # print("The computed probabilities are: ", gate.return_measurement_probabilities())
    # gate.add_cphase()
    # gate.plot_gate_sequence()
    # fid_noise = gate.compute_gate_sequence(0., runs=100)
    # print("The computed fidelity is: ", gate.return_computed_fidelity())
    # gate.plot_computed_mappings()

    