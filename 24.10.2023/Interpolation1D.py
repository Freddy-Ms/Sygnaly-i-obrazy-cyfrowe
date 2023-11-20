import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scp
N = 100
def Sin(x):
    return np.sin(x)

def SinX1(x):
    return np.sin(1/x)

def signum(x):
    return np.sign(np.sin(8*x))

x0 = np.linspace(-np.pi,np.pi,N)
y0 = [Sin(alfa) for alfa in x0]
y0_inverted = [SinX1(alfa) for alfa in x0]
y0_signum = [signum(alfa) for alfa in x0]
_ = plt.plot(x0,y0, 'bo')