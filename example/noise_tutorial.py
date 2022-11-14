from DM_solver.solver  import H_channel, H_solver
from DM_solver.pulse_generation.pulse import pulse
from DM_solver.noise.noise_sources import static_noise_generator, spectral_noise_generator
from DM_solver.utility.pauli import X,Y,Z

import matplotlib.pyplot as plt
import numpy as np

'''
Example 1 : use static noise that decoheres the qubit
'''
f_qubit = 1e9

# define channel for that sets the energy separation -- Sz Hamiltonian. Units of [rad]
Qubit1_Z = H_channel(Z/2)
Qubit1_Z.pulse.add_constant(2*np.pi*f_qubit)

# set T2 of 50ns
Qubit1_Z.add_noise(static_noise_generator(np.sqrt(2)/50e-9))

# make object that solves the Schrodinger equation
calculation = H_solver()
calculation.add_channels(Qubit1_Z)

# initial density matrix (put in superposition)
psi_0 = np.matrix([[1,1],[1,1]])/2

# calculate for 100ns with time steps of 10ps
calculation.calculate(psi_0, end_time = 100e-9, sample_rate = 1e11, n_iter=200)

# calculate some expectation values and plot
Z_expect, X_expect = calculation.return_expectation_values(Z, X)
t = calculation.return_time()
plt.plot(t, Z_expect)
plt.plot(t, X_expect, label = 'expectation value in <X>')
plt.plot(t, np.exp(-(t/50e-9)**2), label = "expectation theory")
plt.legend()
plt.show()

'''
Example 2 : add a lindbladian to simulate T1 (10ns)
'''
f_qubit = 1e9

# define channel for that sets the energy separation -- Sz Hamiltonian. Units of [rad]
Qubit1_Z = H_channel(Z/2)
Qubit1_Z.pulse.add_constant(2*np.pi*f_qubit)

# make object that solves the Schrodinger equation
calculation = H_solver()
calculation.add_channels(Qubit1_Z)
calculation.add_lindbladian(np.matrix([[0,0],[1,0]]), np.sqrt(1/10e-9))
# initial density matrix (put in superposition)
psi_0 = np.matrix([[1,0],[0,0]])

# calculate for 100ns with time steps of 10ps
calculation.calculate(psi_0, end_time = 100e-9, sample_rate = 1e11, n_iter=1)

# calculate some expectation values and plot
Z_expect, X_expect = calculation.return_expectation_values(Z, X)
t = calculation.return_time()
plt.plot(t, Z_expect, label = 'expectation value in <Z>')
plt.plot(t, X_expect, label = 'expectation value in <X>')
plt.plot(t, 2*np.exp(-(t/10.1e-9))-1, label = "expectation theory")
plt.legend()
plt.show()

'''
Example 3 : adding noise originating from a spectrum
'''
f_qubit = 1e9

# define channel for that sets the energy separation -- Sz Hamiltonian. Units of [rad]
Qubit1_Z = H_channel(Z/2)
Qubit1_Z.pulse.add_constant(2*np.pi*f_qubit)

# add noise function
A = 1
OneOver_f_noise = lambda f: A/f
f_cutoff = None #set to value to integrate noise to the cutoff and add it under a static form
# hard to find specific amplitude, best to fit it with methods similar to the ones used in sample 2
Qubit1_Z.add_noise(spectral_noise_generator(OneOver_f_noise, f_cutoff))
Qubit1_Z.plot_pulse(100e-9)

'''
Example 4 : correlate noise -- add correlated noise between two channels
'''
Qubit1_A = H_channel(Z/2)
Qubit1_B = H_channel(X/2)

OneOver_f_noise = lambda f: 1/f
my_noise_source = spectral_noise_generator(1e8, OneOver_f_noise)

# both qubits will undergo the same noise
Qubit1_A.add_noise(0.5*my_noise_source)
# NOTE : multiply by x or copy.copy() is needed, 
# it make sure that both noise sources have the same seed!
Qubit1_B.add_noise(1*my_noise_source)
