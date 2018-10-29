# -*- coding: utf-8 -*-
"""
Created on 

@author: fame
"""

 
from load_mnist import * 
import hw1_linear  as mlBasics  
import numpy as np 


# Read in training and test data
X_train, y_train = load_mnist('training', [0,1]  ) 
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_train = np.divide(X_train, 256)
X_test, y_test = load_mnist('training', [0,1]  )
X_test = np.reshape(X_test, (X_test.shape[0], -1)) 
X_test = np.divide(X_test, 256)


# Starting values for weights W and bias b
W0 = np.zeros(X_train.shape[1])
b0 = 0

# Optimization
num_iters = 100; 
eta = 0.001
W,b = mlBasics.train(X_train, y_train, W0, b0, num_iters, eta)
 

# Test on test data 
yhat = mlBasics.predict(X_test, W, b)>=.5 
print np.mean(yhat==y_test)*100, "% of test examples classified correctly."
  