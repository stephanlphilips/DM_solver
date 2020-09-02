#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 23:23:22 2020

@author: thsiao
"""

#%%
import numpy as np
from scipy import signal
from scipy import interpolate
import os
dir_keysight_ar = os.path.dirname(__file__) # get path to this script in order to load 'frequency_response_650mV_no_ring.csv'
path_keysight_ar = os.path.join(dir_keysight_ar,'frequency_response_650mV_no_ring.csv') # build the full file path
#%%
def keysight_anti_ringing_freq_response():
    '''
    Load and process Keysight anti ringing frequency response

    Returns
    -------
    keysight_freq_array_norm :
        frequency in Hz
    keysight_amp_array_norm : TYPE
        Normalized amplitude response
    keysight_phs_array_norm_filt : TYPE
        Smoothed phase response in rad

    '''
    
    # Load measured frequency response
    # keysight_freq_response = np.genfromtxt('frequency_response_650mV_no_ring.csv',delimiter=',')
    keysight_freq_response = np.genfromtxt(path_keysight_ar,delimiter=',')
    keysight_freq_response = keysight_freq_response.T
    keysight_freq_array = keysight_freq_response[0]
    keysight_amp_array = keysight_freq_response[1]
    keysight_phs_array = keysight_freq_response[2]
    
    # Extropolate the signal to 0-500MHz. Normalize the amplitude response and smooth the phase response
    keysight_freq_array_norm = np.linspace(0,5e8,101)
    keysight_amp_array_norm = np.zeros(len(keysight_freq_array_norm))
    keysight_amp_array_norm[0:3] = keysight_amp_array[1]
    keysight_amp_array_norm[3:95] = keysight_amp_array[1:]
    keysight_amp_array_norm[94:] = np.linspace(keysight_amp_array[-1],0,7)
    keysight_amp_array_norm = keysight_amp_array_norm/keysight_amp_array_norm[0]
    # Add 1/sinc(f/f_sample) correction, which makes the pulse shape more realistic (according to Sander)
    keysight_amp_array_norm = keysight_amp_array_norm/np.sinc(keysight_freq_array_norm/1e9)
    
    f_explt = interpolate.interp1d(keysight_freq_array,keysight_phs_array,fill_value='extrapolate')
    keysight_phs_array_norm = f_explt(keysight_freq_array_norm)
    keysight_phs_array_norm[0:2] = keysight_phs_array[0]
    keysight_phs_array_norm += -keysight_phs_array_norm[0]
    # smooth the phase response
    keysight_phs_array_norm_filt = signal.savgol_filter(keysight_phs_array_norm, 11, 3)
    keysight_phs_array_norm_filt += -keysight_phs_array_norm_filt[0]
    
    return keysight_freq_array_norm, keysight_amp_array_norm, np.pi*(keysight_phs_array_norm_filt/180)
    # return keysight_freq_array_norm/1e9, keysight_amp_array_norm, np.pi*(keysight_phs_array_norm/180)

def keysight_anti_ringing_filter(freq_array_input):
    '''
    Generate an array of complex number for the anti ringing filter

    Parameters
    ----------
    freq_array_input : TYPE
        Frequency input in Hz

    Returns
    -------
    filter_array : TYPE
        Complex array for the anti ringing filter

    '''
    
    keysight_freq, keysight_amp, keysight_phs = keysight_anti_ringing_freq_response()
    
    keysight_freq_neg = np.flip(-1*keysight_freq)
    keysight_amp_neg = np.flip(keysight_amp)
    keysight_phs_neg = np.flip(-1*keysight_phs)
    
    keysight_freq_full = np.concatenate((keysight_freq_neg[0:-1],keysight_freq))
    keysight_amp_full = np.concatenate((keysight_amp_neg[0:-1],keysight_amp))
    keysight_phs_full = np.concatenate((keysight_phs_neg[0:-1],keysight_phs))
    
    # keysight_amp_intp_func = interpolate.interp1d(keysight_freq, keysight_amp, bounds_error = False ,fill_value = (0,0))
    # keysight_phs_intp_func = interpolate.interp1d(keysight_freq, keysight_phs, bounds_error = False ,fill_value = (0,np.pi/2))
    
    keysight_amp_intp_func = interpolate.interp1d(keysight_freq_full, keysight_amp_full, bounds_error = False ,fill_value = (0,0))
    keysight_phs_intp_func = interpolate.interp1d(keysight_freq_full, keysight_phs_full, bounds_error = False ,fill_value = (np.pi/2,np.pi/2))
    
    keysight_amp_intp_array = keysight_amp_intp_func(freq_array_input)
    keysight_phs_intp_array = keysight_phs_intp_func(freq_array_input)
    
    filter_array = keysight_amp_intp_array*np.exp(1j*keysight_phs_intp_array)
    
    return filter_array

def keysight_anti_ringing_filtered_output(input_signal, sampling_rate):
    '''
    Filter the input signal using the measured Keysight anti ringing frequency response

    Parameters
    ----------
    input_signal : TYPE
        Input signal.
    sampling_rate : TYPE
        Sampling rate in Hz.

    Returns
    -------
    output_signal : TYPE
        Filtered signal

    '''
    
    extra_signal_time = 200e-9 # add 100ns flat signal before and after the input signal to avoid start/end issue
    extra_signal_1 = np.ones(int(extra_signal_time*sampling_rate))*input_signal[0] # extra signal before the true input signal
    extra_signal_2 = np.ones(int(extra_signal_time*sampling_rate))*input_signal[-1] # extra signal before the true input signal
    
    input_signal_extra = np.concatenate((extra_signal_1,input_signal,extra_signal_2))
    
    freq_array = np.fft.fftfreq(len(input_signal_extra),1/sampling_rate)
    input_signal_fft = np.fft.fft(input_signal_extra)
    filter_array = keysight_anti_ringing_filter(freq_array)
    
    input_signal_fft_filt = input_signal_fft*filter_array
    
    input_signal_fft_ifft = np.fft.ifft(input_signal_fft_filt)
    
    output_signal = input_signal_fft_ifft[len(extra_signal_1):-len(extra_signal_2)]
    
    return output_signal
