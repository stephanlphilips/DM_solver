import numpy as np
import matplotlib.pyplot as plt

def return_STD_omega_for_sim(spectrum, sample_rate, n_points):
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

def mk_noise(STD_omega, n_points):
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

if __name__ == '__main__':

    one_over_f_noise = lambda omega: 2*np.pi/omega

    # cut off made to compare if the scaling in the algorithm worked as expected.
    # def one_over_f_noise(omega):
    #     S = 2*np.pi/omega
    #     S[np.where(omega>1e6)[0]]=0
    #     return S

    npt = 1048576 #fastest for 2**n (e.g. try 1000001 for a slow fft)
    sample_rate = 1e9
    S = return_STD_omega_for_sim(one_over_f_noise, sample_rate, npt)
    noise1 = mk_noise(S, npt)
    t1 = np.linspace(0,npt/sample_rate,npt)

    npt = 1024
    sample_rate = 1e6
    S = return_STD_omega_for_sim(one_over_f_noise, sample_rate, npt)
    noise2 = mk_noise(S, npt)
    t2 = np.linspace(0,npt/sample_rate,npt)

    plt.plot(t1,noise1)
    plt.plot(t2,noise2)
    

    # timeinterval =20
    # sampleSize=timeinterval*1500
    # test_noise = NoiseGenerator(oneoverfnoise,sampleSize,timeinterval)
        # optionally plot the Power Spectral Density with Matplotlib
    from matplotlib import mlab
    from matplotlib import pylab as plt
    s, f = mlab.psd(noise1, NFFT=noise1.size)
    plt.figure()
    plt.loglog(f,s)
    plt.grid(True)
    plt.show()