import math as m
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def f(X):
    return np.array([m.sin(i/5)*m.exp(i/10)+5*m.exp(-i/2) for i in X])

def draw():
    X = np.arange(40)
    plt.plot(X, f(X))
    plt.show()

x0 = 30.
res = minimize(f, x0, method = 'BFGS',options={'disp': True})
print (res)

draw()

