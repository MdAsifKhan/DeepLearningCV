# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""

import numpy as np 
from matplotlib import pyplot as plt
import pdb

def predict(X,W,b):  
    """
    implement the function h(x, W, b) here  
    X: N-by-D array of training data 
    W: D dimensional array of weights
    b: scalar bias

    Should return a N dimensional array  
    """
    return sigmoid(np.dot(X, W) + b)   
 
def sigmoid(a): 
    """
    implement the sigmoid here
    a: N dimensional numpy array

    Should return a N dimensional array  
    """
    return 1/(1+np.exp(-a))

def l2loss(X,y,W,b):  
    """
    implement the L2 loss function
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias

    Should return three variables: (i) the l2 loss: scalar, (ii) the gradient with respect to W, (iii) the gradient with respect to b
     """
    pred = predict(X, W, b)
    pdb.set_trace()
    l2 = np.sum(np.square(y-pred))
    gradE_pred = y - pred
    gradE_h = gradE_pred*pred*(1-pred)
    gradE_h = gradE_h.reshape((gradE_h.shape[0],1))
    gradE_W = np.mean((-2*X)*gradE_h, axis=0)
    gradE_b = np.mean(-2*gradE_h, axis=0)
    return l2, gradE_W, gradE_b

def train(X,y,W,b, num_iters=1000, eta=0.001):  
    """
    implement the gradient descent here
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias
    num_iters: (optional) number of steps to take when optimizing
    eta: (optional)  the stepsize for the gradient descent

    Should return the final values of W and b    
     """
    loss = []
    eta = 1
    for i in range(num_iters):
        l2, gradE_W, gradE_b = l2loss(X, y, W, b)

        W = W - eta*gradE_W 
        b = b - eta*gradE_b

        loss.append(l2)
    plt.plot(range(num_iters), loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    return W, b
 