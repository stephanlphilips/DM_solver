import numpy as np
from qutip import *
from inspect import getfullargspec
import cmath
from scipy.optimize import curve_fit
# from lmfit import *
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from numpy import concatenate, real, std, abs
from numpy.fft import ifft, fftfreq
from numpy.random import normal
from matplotlib import *
from pylab import *
import scipy.integrate as integrate
from scipy.interpolate import *


oneoverfnoise=lambda f: 1/f

# def NoiseGenerator(spectrum, samples, totaltime, fmin=0.):
#     f = fftfreq(int(samples))*samples/totaltime
#     s_scale = abs(concatenate([f[f<0], [f[-1]]]))
#     print(s_scale)
#     idx = np.where(s_scale < fmin)[0]
#     print(idx)
#     s_scale = np.sqrt(spectrum(s_scale)*samples/totaltime)    
#     sr = s_scale * normal(size=len(s_scale))
#     si = s_scale * normal(size=len(s_scale))
#     if not (samples % 2): si[0] = si[0].real
#     s = sr + 1J * si
#     s = concatenate([s[1-int(samples % 2):][::-1], s[:-1].conj()])
#     s[idx] = 0

#     y = ifft(s).real

#     return y / std(y)

# # noise = NoiseGenerator(oneoverfnoise, 1e4, 100e-9, 1e9)


# timeinterval =1e-9

# sampleSize=int(1e-6/timeinterval)

# test_noise = NoiseGenerator(oneoverfnoise,sampleSize,timeinterval, 0.1)

#               # optionally plot the Power Spectral Density with Matplotlib

# from matplotlib import mlab

# from matplotlib import pylab as pltx

# s, f = mlab.psd(test_noise, NFFT=sampleSize)

# plt.loglog(f,s)

# plt.grid(True)
# plt.show()

if not (6%2):
    print(True, 6%2)