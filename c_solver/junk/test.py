import numpy as np

STD_omega_sqrt = 500

# d1 = STD_omega_sqrt*np.random.normal(size=10000)
# d2 = np.random.normal(scale=STD_omega_sqrt,size=10000)


# import matplotlib.pyplot as plt

# _ = plt.hist(d1, bins='auto')
# _ = plt.hist(d2, bins='auto')
# plt.show()

print(np.fft.fftfreq(101,0.99))