import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0,100000e-9,100000)
w1 = 50e6
w2 = 250e6

A = 1
c = 10

def exchange(x):
	return 3.31980207e+04*np.exp(1.47402607e-01*x)

sig1 = 1e-6*exchange(38*np.sin(w1*time*2*np.pi))
sig2 = 1e-6*exchange(38*np.sin(w2*time*2*np.pi))
# sig1 = np.sin(w1*time*2*np.pi)
sig = sig1 + sig2

# plt.plot(time, sig1)
# plt.show()


sp = np.fft.fft(sig1)
freq = np.fft.fftfreq(time.shape[0], time[1]-time[0])
plt.plot(freq*1e-6, sp.real/np.max(sp.real))
plt.xlabel("Frequency (MHz)")
plt.ylabel("Amplitude (a.u.)")

# plt.figure()
# plt.plot(freq*1e-6, sp.imag)

plt.show()