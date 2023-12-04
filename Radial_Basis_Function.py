import matplotlib.pyplot as plt
import numpy as np
from random import random
from math import pi, sin, tanh, exp

# 1. Data preparation
x = np.arange(0, 1, 1/22)
print(x)
print(type(x))
print()

d = []
for xn in x:
    d.append(((1 + 0.6 * sin (2 * pi * xn / 0.7)) + 0.3 * sin (2 * pi * xn)) / 2)
print(d)

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.plot(x, d, label="Desired output")

# 2. Strucuture selection
# one hidden layer with two Gaussian radial basis functions neurons, one output neuron

# 3. Initiate parameters
w1 = random(); w2 = random()
b = random()
c1 = 0.18  #perfect value: 0.18
r1 = 0.15  #perfect value: 0.15
c2 = 0.9   #perfect value: 0.9
r2 = 0.175 #perfect value: 0.175
eta = 0.01

test = np.zeros(len(x))
for k in range(1000):
    for i in range(len(x)):
        v1 = exp(-((x[i]-c1)**2)/(2*r1**2))
        v2 = exp(-((x[i]-c2)**2)/(2*r2**2))

        y = w1*v1 + w2*v2 + b
        test[i] = y
        
        e = d[i] - y

        w1 = w1 + eta * e * v1
        w2 = w2 + eta * e * v2
        b  = b + eta * e
##
##ax.plot(x, test, label="Radial Basis Function test")
##ax.set_title("Radial Basis Function")
##ax.legend(loc = "lower left")
##plt.show()

# Testing
X = np.arange(0, 1, 1/220)
Y = np.zeros(len(X))
for i in range(len(X)):
    v1 = exp(-((X[i]-c1)**2)/(2*r1**2))
    v2 = exp(-((X[i]-c2)**2)/(2*r2**2))

    Y[i] = w1 * v1 + w2 * v2 + b

print()
print(Y)
ax.plot(X, Y, label="Radial Basis Function output")
ax.set_title("Radial Basis Function")
ax.legend(loc = "lower left")
plt.show()









    
