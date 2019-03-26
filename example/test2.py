import scipy.optimize
import numpy as np
x = np.array([0,25,35,40])
y = np.array([0,1,6,12])*1e6

def func(x,a,b):
	return a*np.exp(b*x)
popt, pcov = scipy.optimize.curve_fit(func,  x,  y, p0 = [0.1e6,1/8])

print(popt)
import matplotlib.pyplot as plt

plt.plot(x, y)
x = np.linspace(0, 40,100)
plt.plot(x, func(x, *popt))
# plt.plot(x, 0.1e6*(np.exp(x/8.5)-1))
plt.show()


