import math as m
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def f(X):
    return [m.sin(i/5)*m.exp(i/10)+5*m.exp(-i/2) for i in X]

def slau(X, Y):
    A = np.array([X**i for i in range(X.size)])
    A = np.transpose(A)
    return (linalg.solve(A, Y))

def get_pol(W, x):
    k = 0
    y = 0
    for i in W:
        y += i * x ** k
        k += 1
    return y
        

def pol(W,X):
    return ([get_pol(W, i) for i in X])
    

def draw(W):
    X = np.arange(16)
    plt.plot(X, f(X), pol(W, X))
    plt.show()

X = np.array([1,4, 10, 15])
Y = f(X)
W = slau(X,Y)
print(W)
draw(W)