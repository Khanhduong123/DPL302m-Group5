from planar_utils import *
import numpy as np
import copy
import matplotlib.pyplot as plt
from testCases_v2 import *
from public_tests import *
import sklearn
import sklearn.datasets
import sklearn.linear_model


def layer_size(X,Y):
    """
    X: input dataset of shape
    Y: label of shape

    return 
    n_x: size of the input layer
    n_h: size of the hidden layer
    n_y: size of the output layer
    """

    n_x= X.shape[0]
    n_h = 4
    n_y= Y.shape[0]
    return (n_x,n_h,n_y)


def initialize_parameters(n_x,n_h,n_y):
    """
    n_x : size of the input layer
    n_h : size of the hidden layer
    n_y : size of the output layer

    return parameters
    W1 -- weight matrix of shape (n_h, n_x)
    b1 -- bias vector of shape (n_h, 1)
    W2 -- weight matrix of shape (n_y, n_h)
    b2 -- bias vector of shape (n_y, 1)
    """

    W1= np.random.randn(n_h,n_x) *0.01
    b1= np.zeros(shape=(n_h,1))
    W2= np.random.randn(n_y,n_h) * 0.01
    b2= np.zeros(shape=(n_y,1))

    parameters={'W1':W1,
               'b1':b1,
               'W2':W2,
               'b2':b2}
    return parameters

def forward_propagate(X, parameters):
    """
    X: input data shape(n_x,m)
    parameters: output initialize_parameters

    return
    A2: the sigmoid of output layer 2
    cache: a dictionary save the value (W1,b1,W2,b2)
    """
    W1= parameters['W1']
    b1= parameters['b1']
    W2= parameters['W2']
    b2= parameters['b2']

    #forward propagate
    Z1= np.dot(W1,X) + b1
    A1= np.tanh(Z1)
    Z2= np.dot(W2,A1) + b2
    A2= sigmoid(Z2)

    cache= {'Z1':Z1,
            'A1':A1,
            'Z2':Z2,
            'A2':A2}
    return A2, cache


def compute_cost(A2,Y):
    """
    Input
    A2: the predict vector values
    Y: the actual vector values

    return
    cost: cross-entropy cost given equation (13)
    """
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2),Y) + np.multiply((1-Y),np.log(1-A2))
    cost = - np.sum(logprobs) / m
    cost = float(np.squeeze(cost))
    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m 
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / m 
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m 
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    #retrive parameters 
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    #retrive derative parameters
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    #update parameters using gradient descent
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    #list of parameters after update
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    A2, cache = forward_propagate(X, parameters)
    predictions = np.round(A2)
    return predictions

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        A2, cache = forward_propagate(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    return parameters