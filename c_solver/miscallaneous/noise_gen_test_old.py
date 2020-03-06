import numpy as np
import matplotlib.pyplot as plt

def return_spectral_density_for_sim(spectrum, sample_rate, n_points):
    '''
    Args :
        spectrum (lambda) : function that gives the spectral density for given input freq in Hz.
        sample_rate (double) : sample rate of the experiment.
        n_points (int) : number of points of noise to generate.
    
    Returns :
        spectral density at the frequencies relevant for the simulation. 
    '''

    frequencies = np.fft.fftfreq(n_points)*sample_rate

    # get postive frequencies (we will be taking the sqrt as we will add up real and imag components)
    freq_postive = abs(np.concatenate([frequencies[frequencies<0], [frequencies[-1]]]))
    S_omega_sqrt = np.sqrt(spectrum(freq_postive))*sample_rate    

    return S_omega_sqrt

def mk_noise(S_omega_sqrt, sample_rate, n_points, A):
    '''
    test function to generate, will be ported to c++
    '''

    # randomise the amound of noise on each of the freq components(also imag to randomize the phase)
    S_omega_real = S_omega_sqrt * np.random.normal(size=S_omega_sqrt.size)
    S_omega_imag = S_omega_sqrt * np.random.normal(size=S_omega_sqrt.size)
    
    # Reconstruct in the right formatting to take a inverse fourrier transform
    if not (n_points % 2): S_omega_imag[0] = S_omega_imag[0].real
    S_omega = S_omega_real + 1J * S_omega_imag
    S_omega = np.concatenate([S_omega[1-int(n_points % 2):][::-1], S_omega[:-1].conj()])
    S_omega[0] = 0
    # get real values to get the noise
    noise_values = np.fft.ifft(S_omega).real

    return noise_values*A

if __name__ == '__main__':

    def oneoverfnoise(omega):
        S = 2*np.pi/omega
        S[np.where(omega>1e6)[0]]=0
        return S

    npt = 1000000
    sample_rate = 1e9
    S = return_spectral_density_for_sim(oneoverfnoise, sample_rate, npt)
    noise1 = mk_noise(S,sample_rate, npt,1)
    t1 = np.linspace(0,npt/sample_rate,npt)

    npt = 1000
    sample_rate = 1e6
    S = return_spectral_density_for_sim(oneoverfnoise, sample_rate, npt)
    noise2 = mk_noise(S,sample_rate, npt,1)
    t2 = np.linspace(0,npt/sample_rate,npt)

    plt.plot(t1,noise1)
    plt.plot(t2,noise2)
    plt.show()

# def NoiseGenerator(spectrum, samples, totaltime, fmin=0.):
#     f = fftfreq(int(samples))*samples/totaltime
#     print(f)
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

# noise = NoiseGenerator(oneoverfnoise, 1e3, 1e-6)

# plt.plot(noise)
# plt.show()
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
