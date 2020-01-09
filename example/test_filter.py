#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:44:55 2019

@author: mruss
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

bFIR = signal.firwin(21, 400*1e6, pass_zero = 'lowpass',fs=1e10)
bIIR, aIIR = signal.butter(3, 400*1e6, btype = 'lowpass',fs=1e10)


t = np.linspace(0, 100*1e-9, 1001)
delay=20*1e-9
def enevlope_fun(time):
            return (np.arctan(3)+np.arctan(6*(time-delay/2)/delay))/(2*np.arctan(3))
    
xn = enevlope_fun(t)

wFIR = signal.lfilter(bFIR,[1.0],xn)
w1, h1 = signal.freqz(bFIR,fs=1e10)
wIIR = signal.lfilter(bIIR,aIIR,xn)
w2, h2 = signal.freqz(bIIR,aIIR,fs=1e10)

plt.title('Digital filter frequency response')
plt.plot(w1, 20*np.log10(np.abs(h1)), 'r')
plt.plot(w2, 20*np.log10(np.abs(h2)), 'g')
plt.xlim(0,1e9)
plt.ylabel('Amplitude Response (dB)')
plt.xlabel('Frequency (rad/sample)')
plt.grid()
plt.show()

plt.title('Digital filter frequency response')
plt.plot(t, xn, 'b')
plt.plot(t, wFIR, 'r')
plt.plot(t, wIIR, 'g')
plt.xlim(0, 5*delay)
plt.ylim(0, 1)
plt.ylabel('Amplitude Response (dB)')
plt.xlabel('Frequency (rad/sample)')
plt.grid()
plt.show()