# -*- coding: utf-8 -*-
"""
Created on 

@author: fame
"""

 
from load_mnist import * 
import hw1_knn  as mlBasics  
import numpy as np 
   
  
# Load data - two class 
X_train, y_train = load_mnist('training' , [0,1] )
X_test, y_test = load_mnist('testing'  ,  [0,1]  )

# Load data - ALL CLASSES
#X_train, y_train = load_mnist('training'  )
#X_test, y_test = load_mnist('testing'   )


 # Reshape the image data into rows  
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

  
# Test on test data   
#1) Compute distances:
dists =  mlBasics.compute_euclidean_distances(X_train,X_test) 
  
#2) Run the code below and predict labels: 
y_test_pred = mlBasics.predict_labels(dists, y_train ) 

#3) Report results
# you should get following message '99.91 of test examples classified correctly.'
print('{0:0.02f}'.format(  np.mean(y_test_pred==y_test)*100), "of test examples classified correctly.")

  
