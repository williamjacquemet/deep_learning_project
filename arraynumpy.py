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
        return x * (1 - x)
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

n=5000
m=50000
eta = 0.02
w= [0.3,0.5,1]
mean = [0, 0]
cov = [[0.1, 0], [0, 0.1]]  # diagonal covariance
mean1 = [2, 2]
cov1 = [[0.1, 0], [0, 0.1]]  # diagonal covariance
bias= []
for i in range(n):
    bias.append(1)

bias= np.hstack(bias)
rand=randint(0, n)
x1,y1 = np.random.multivariate_normal(mean1, cov1, n).T # les 1
x,y = np.random.multivariate_normal(mean, cov, n).T# les 0
x=np.reshape(x, (n,1))
x1=np.reshape(x1, (n,1))
y=np.reshape(y, (n,1))
y1=np.reshape(y1, (n,1))
bias =np.reshape(bias,(n,1))


b = np.concatenate((x,y,bias), axis =1)
a = np.concatenate((x1,y1,bias), axis =1)



for i in range(m):
    rand=randint(0, n)
    if(rand<(n/2)):
        randomvalueofx = choice(b)
        result= sigmoid(dot(w,randomvalueofx))
        error = 0 - result       
        w += eta * error * sigmoid_derivative(result)* randomvalueofx
        
    else:
        randomvalueofx1y1 = choice(a)
        result= sigmoid(dot(w,randomvalueofx1y1))
        error = 1 - result       
        w += eta * error * sigmoid_derivative(result)* randomvalueofx1y1
        
print("w après",w,"valeur1",w[0])
pt2=-w[2]/w[1]
pt1=-w[2]/w[0]
print(pt1)
plt.plot([pt2,0],[0,pt1])
plt.plot(x,y, 'x')
plt.plot(x1, y1, 'g^')
plt.axis('equal')
plt.show()
