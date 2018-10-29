# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""
 
import numpy as np 
 

def compute_euclidean_distances( X, Y ) :
    """
    Compute the Euclidean distance between two matricess X and Y  
    Input:
    X: N-by-D numpy array 
    Y: M-by-D numpy array 
    
    Should return dist: M-by-N numpy array   
    """  
    return np.array([[np.sqrt(np.sum(np.power(x-y,2))) for x in X] for y in Y])

def predict_labels( dists, labels, k=1):
    """
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array 
    labels: is a N dimensional numpy array
    
    Should return  pred_labels: M dimensional numpy array
    """
    pred_labels = np.zeros(len(dists))
    for i,dist in enumerate(dists):
        nearest_neighbor = labels[dist.argsort()[:k]]
        predictions, count = np.unique(nearest_neighbor, return_counts=True)
        # Most frequent value
        pred_labels[i] = predictions[np.argmax(count)]

    return pred_labels