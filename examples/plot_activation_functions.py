import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def gelu(x):
   return x * norm.cdf(x)

def relu(x):
   return np.maximum(0, x)

def swish(x, beta=1):
   return x * (1 / (1 + np.exp(-beta * x)))

x_values = np.linspace(-5, 5, 500)
sigmoid_values = sigmoid(x_values)
linear_values = x_values
gelu_values = gelu(x_values)
relu_values = relu(x_values)
swish_values = swish(x_values)

fig = plt.figure()
plt.plot(x_values, sigmoid_values, label='sigmoid')
plt.plot(x_values, linear_values, label='linear')
plt.plot(x_values, gelu_values, label='GELU')
plt.plot(x_values, relu_values, label='ReLU')
plt.plot(x_values, swish_values, label='Swish')
plt.title("Activation Functions")
plt.xlabel("x")
plt.ylabel("Activation")
plt.grid()
plt.legend()
plt.show()

'''
############################ LIST OF AMENDMENTS ##########################################

This file is entirely written by our team.

##########################################################################################
'''