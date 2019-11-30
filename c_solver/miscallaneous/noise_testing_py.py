import noise_testing
import numpy as np
import matplotlib.pyplot as plt

def get_fft_components(spectrum, sample_rate, n_points):
    '''
    Args :
        n_points (int) : number of points of noise to generate.
        sample_rate (double) : sample rate of the experiment.
    
    Returns :
        STD_omega (np.ndarray<double>) : standard deviations of the noise at the requested frequencies (freq_postive). Normalized by sample frequency.
    '''

    n_points = 2**(int(np.log2(n_points))+1)
    frequencies = np.fft.fftfreq(n_points, d=1/sample_rate)

    # get postive frequencies (we will be taking the sqrt as we will add up real and imag components)
    freq_postive = abs(frequencies[frequencies<0])[::-1]
    STD_omega = np.sqrt(spectrum(freq_postive*2*np.pi))*sample_rate

    print("size of standard deviation", STD_omega.size)
    return STD_omega

def return_STD_omega_for_sim1(spectrum, sample_rate, n_points):
    '''
    Args :
        spectrum (lambda) : function that gives the spectral density for given input given as anguler freq [rad].
        sample_rate (double) : sample rate of the experiment.
        n_points (int) : number of points of noise to generate.
    
    Returns :
        STD_omega (np.ndarray<double>) : standard deviations of the noise at the requested frequencies (freq_postive). Normalized by sample frequency.
    '''

    frequencies = np.fft.fftfreq(n_points, d=1/sample_rate)

    # get postive frequencies (we will be taking the sqrt as we will add up real and imag components)
    freq_postive = abs(frequencies[frequencies<0])[::-1]
    STD_omega = np.sqrt(spectrum(freq_postive*2*np.pi))*sample_rate

    return STD_omega

def mk_noise1(STD_omega, n_points):
    '''
    test function to generate noise, will be ported to c++

    Args:
        STD_omega (np.ndarray<double>) : standard deviations of the noise at the requested frequencies (freq_postive). Normalized by sample frequency.
        n_points (int) : number of points of noise to generate.

    Return:
        noise_values (np.ndarray<double>) : noise string generated for the given STD_omega
    '''

    # sample noise for all the freq in STD_omega and given standard deviation, also randomize the phase with uniform distribution
    episilon_omega = 0.5*STD_omega * np.random.normal(size=STD_omega.size) *\
                        np.exp(1j*np.random.uniform(0,2*np.pi,STD_omega.size))
    
    # reserve memory
    episilon_f_FFT = np.zeros(n_points, dtype=np.complex)

    if n_points%2 == 0: #even        
        episilon_f_FFT[1:STD_omega.size] = episilon_omega[:-1]
        episilon_f_FFT[STD_omega.size:] = episilon_omega.conj()[::-1]
        # correct for freq not being 0, see docs
        episilon_f_FFT[STD_omega.size] = 0
    else: #odd
        episilon_f_FFT[1:STD_omega.size+1] = episilon_omega
        episilon_f_FFT[STD_omega.size+1:] = episilon_omega.conj()[::-1]
    
    # get effective noise values (cast to double data type)
    noise_values = np.fft.ifft(episilon_f_FFT).real

    return noise_values

def return_STD_omega_for_sim2(spectrum, sample_rate, n_points):
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

def mk_noise2(S_omega_sqrt, n_points, A=1):
    '''
    test function to generate, will be ported to c++
    '''

    # randomise the amound of noise on each of the freq components(also imag to randomize the phase)
    S_omega_real = 0.5*S_omega_sqrt * np.random.normal(size=S_omega_sqrt.size)
    S_omega_imag = 0.5*S_omega_sqrt * np.random.normal(size=S_omega_sqrt.size)
    
    # Reconstruct in the right formatting to take a inverse fourrier transform
    if not (n_points % 2): S_omega_imag[0] = S_omega_imag[0].real
    S_omega = S_omega_real + 1J * S_omega_imag
    S_omega = np.concatenate([S_omega[1-int(n_points % 2):][::-1], S_omega[:-1].conj()])
    S_omega[0] = 0
    # get real values to get the noise
    noise_values = np.fft.ifft(S_omega).real

    return noise_values*A/np.sqrt(2)

one_over_f_noise = lambda omega: 2*np.pi/omega

npt = 1000000 #fastest for 2**n (e.g. try 1000001 for a slow fft)
sample_rate = 1e9
S = get_fft_components(one_over_f_noise, sample_rate, npt)
noise1 = noise_testing.return_noise(S, npt)
t1 = np.linspace(0,npt/sample_rate,npt)

npt = 1000000
sample_rate = 1e9
S = return_STD_omega_for_sim1(one_over_f_noise, sample_rate, npt)
noise2 = mk_noise1(S, npt)
t2 = np.linspace(0,npt/sample_rate,npt)

plt.plot(t1,noise1)
plt.plot(t2,noise2)

plt.show()

from matplotlib import mlab
from matplotlib import pylab as plt
s, f = mlab.psd(noise1, NFFT=noise1.size)
s2, f2 = mlab.psd(noise2, NFFT=noise2.size)

plt.figure()
plt.loglog(f,s)
plt.loglog(f2,s2)
plt.grid(True)
plt.show()