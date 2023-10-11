import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import scipy.io
from reg_ultils_2 import *
from testCases import *
from public_tests import *


def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    This function compute cost with regularization
    
    Inputs:
    A3 : post-activation, output of forward propagation, of shape (output size, number of examples)
    Y : "true" labels vector, of shape (output size, number of examples)
    parameters : python dictionary containing parameters of the model
    
    Outputs:
    cost : value of the regularized loss function 
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    #compute loss function
    cross_entropy_cost = compute_cost(A3, Y) 
    # compute L2_regularization_cost
    L2_regularization_cost = (1/m) * (lambd/2) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.
    
    Arguments:
    X : input dataset, of shape (input size, number of examples)
    Y : "true" labels vector, of shape (output size, number of examples)
    cache : cache output from forward_propagation()
    lambd : regularization hyperparameter, scalar
    
    Returns:
    gradients : A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y

    dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd/m) * W3
    db3 = 1./ m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    

    dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd/m) * W2
    db2 = 1./ m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    

    dW1 = 1./m * np.dot(dZ1, X.T) + (lambd/m) * W1
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients


def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    This function build the forward propagation with drop_out: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Inputs:
    X : input dataset, of shape (2, number of examples)
    parameters : python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 : weight matrix of shape (20, 2)
                    b1 : bias vector of shape (20, 1)
                    W2 : weight matrix of shape (3, 20)
                    b2 : bias vector of shape (3, 1)
                    W3 : weight matrix of shape (1, 3)
                    b3 : bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Outputs:
    A3 : last activation value, output of the forward propagation, of shape (1,1)
    cache : tuple, information stored for computing the backward propagation
    """
    
    np.random.seed(1)
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
         
    D1 = np.random.rand(A1.shape[0], A1.shape[1])  
    D1 = (D1 < keep_prob).astype(int)              
    A1 = A1*D1                                     
    A1 = A1/keep_prob                              

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    
    
    D2 = np.random.rand(A2.shape[0], A2.shape[1])  
    D2 = (D2 < keep_prob).astype(int)              
    A2 = A2*D2                                     
    A2 = A2/keep_prob                              

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    This function is built the backward propagation of our baseline model to which added dropout.
    
    Inputs:
    X : input dataset, of shape (2, number of examples)
    Y : "true" labels vector, of shape (output size, number of examples)
    cache : cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Outputs:
    gradients : A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
   
    dA2 = dA2*D2        
    dA2 = dA2/keep_prob 

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    
    dA1 = dA1*D1        
    dA1 = dA1/keep_prob 

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients