# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:27:18 2019

@author: william
"""

from random import choice, randint
from numpy import array, dot, random, hstack, newaxis, vstack, repeat
from pylab import plot, ylim

import math
import matplotlib.pyplot as plt
import numpy as np
def sigmoid_derivative(x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
n=2
eta = 0.2
m = 2
w= array([[0.3],[0.5],[1]])
mean = [0, 0]
cov = [[0.1, 0], [0, 0.1]]  # diagonal covariance
mean1 = [2, 2]
cov1 = [[0.1, 0], [0, 0.1]]  # diagonal covariance
bias= []
for i in range(n):
    bias.append(1)

bias= np.hstack(bias)
print("bias",bias)
r=random.rand(1)
rand=randint(0, n)
rand1=randint(0, n)

#x1, y1 = np.random.multivariate_normal(mean1, cov1, n).T # les 1
#x, y = np.random.multivariate_normal(mean, cov, n).T# les 0
pt1 = np.random.multivariate_normal(mean1, cov1, n).T # les 1
pt2 = np.random.multivariate_normal(mean, cov, n).T
#pt1=np.hstack((pt1))
#for i in range(n):
#     pt1.__iadd__(pt1[i])
pt1=pt1.tolist()
w=np.hstack((w))
pt1=np.reshape(pt1, (2,2))
print(pt1,"after 2")
print("test 3emevaleur",pt1[1])

plt.plot(x, y, 'x')
plt.plot(x1, y1, 'g^')
plt.axis('equal')
plt.show()
print(pt1.shape)