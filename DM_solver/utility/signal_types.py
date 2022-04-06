import numpy as np

NORMAL   = lambda x : x
EXP      = lambda x : np.exp(2*x)
EXPSAT   = lambda x : np.pow(np.sqrt(1+exp(-2*(x + log(2.0)))) - exp(-1*(x +log(2))),2)
TANH     = lambda x : np.tanh(x)
SWO1     = lambda x : 1/(1 - np.pow(x,2))
SWO2     = lambda x : (1 + pow(x ,2))/pow(1 - pow(x ,2),3)