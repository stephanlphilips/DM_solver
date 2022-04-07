from DM_solver.solver  import H_channel, H_solver
from DM_solver.pulse_generation.pulse import pulse

from DM_solver.utility.pauli import X,Y,Z

import matplotlib.pyplot as plt
import numpy as np

'''
Example 1 : set zeeman difference and drive qubit
'''
f_qubit = 1e9
f_drive = 10e6

# define channel for that sets the energy separation -- Sz hamiltonian. Units of [rad]
Qubit1_Z = H_channel(Z/2)
Qubit1_Z.pulse.add_constant(2*np.pi*f_qubit)

# define channel that drives the qubit -- Sx hamiltonian.
Qubit1_X = H_channel(X/2)
# define a pulse from 10 ns to 60ns
Qubit1_X.pulse.add_MW_pulse(20e-9,70e-9, amp=2*np.pi*f_drive, freq=f_qubit, phase=np.pi/2)

# show the pulse
Qubit1_X.plot_pulse(t_end=100e-9, sample_rate=1e11)

# make object that solves the schrodinger equation
calculation = H_solver()
calculation.add_channels(Qubit1_Z, Qubit1_X)

# initial density matrix
psi_0 = np.matrix([[1,0],[0,0]])

# calculate for 100ns with time steps of 10ps
calculation.calculate(psi_0, end_time = 100e-9, sample_rate = 1e11)

# calculate some expectatoin values and plot
Z_expect, X_expect = calculation.return_expectation_values(Z, X)
t = calculation.return_time()
plt.plot(t, Z_expect)
plt.plot(t, X_expect)
plt.show()
'''
Example 2 : a pulse with various shapes
'''
test_channel = H_channel(X/2)

test_channel.pulse.add_block(5e-9, 10e-9, 1)
test_channel.pulse.add_ramp( 10e-9, 15e-9, 1, 0)

test_channel.pulse.add_MW_pulse(20e-9, 30e-9, 1, 1e9, AM='blackman')

SineShape = lambda x:(1-np.cos(x*np.pi*2))/2
test_channel.pulse.add_function(35e-9, 45e-9, SineShape)

test_channel.plot_pulse(50e-9)

'''
Example 3 : apply a filter function on a qubit channel
'''
from DM_solver.pulse_generation.filters import keysight_anti_ringing_filtered_output
test_channel.pulse.add_filter(keysight_anti_ringing_filtered_output)
test_channel.plot_pulse(50e-9)


'''
Example 4 : apply standard filter function on a qubit channel
'''
from DM_solver.pulse_generation.filters import keysight_anti_ringing_filtered_output
import scipy.signal as signal

cutoff_freq = 300e6
def scipy_filter(data, sample_rate):
	b, a = signal.butter(3, cutoff_freq/(sample_rate/2))
	return signal.filtfilt(b, a, data)

test_channel = H_channel(X/2)

test_channel.pulse.add_block(5e-9, 10e-9, 1)
test_channel.pulse.add_ramp( 10e-9, 15e-9, 1, 0)
test_channel.plot_pulse(50e-9)
test_channel.pulse.add_filter(scipy_filter)
test_channel.plot_pulse(50e-9)

'''
Example 5 : scale the pulse (+ noise if present)
'''
from DM_solver.utility.signal_types import EXP

test_channel = H_channel(X/2)

test_channel.pulse.add_block(5e-9, 10e-9, 1)
test_channel.pulse.add_ramp( 10e-9, 15e-9, 1, 0)

test_channel.add_pulse_modulator(EXP)

test_channel.plot_pulse(50e-9)