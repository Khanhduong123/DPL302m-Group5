import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets
from opt_utils_v1a import *
from copy import deepcopy
from testCases import *
from public_tests import *


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    This function is updated parameters using one step of gradient descent
    
    Inputs:
    parameters : python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads : python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate : the learning rate, scalar.
    
    Outputs:
    parameters : python dictionary containing your updated parameters 
    """
    L = len(parameters) // 2 # number of layers in the neural networks

    # Update rule for each parameter
    for l in range(1, L + 1):
        
        parameters["W" + str(l)] = parameters['W' + str(l)] - learning_rate * grads["dW" + str(l)] 
        parameters["b" + str(l)] = parameters['b' + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    this function is creates a list of random minibatches from (X, Y)
    
    Inputs:
    X : input data, of shape (input size, number of examples)
    Y : true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size : size of the mini-batches, integer
    
    Outputs:
    mini_batches : list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            
    m = X.shape[1]                  
    mini_batches = []
        
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    
    inc = mini_batch_size

    
    num_complete_minibatches = math.floor(m / mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        
        mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size] 
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Inputs:
    parameters : python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Outputs:
    v : python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(1, L + 1):
        
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        
    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    this function is updated parameters using Momentum
    
    Inputs:
    parameters : python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads : python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v : python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta : the momentum hyperparameter, scalar
    learning_rate : the learning rate, scalar
    
    Outputs:
    parameters : python dictionary containing your updated parameters 
    v : python dictionary containing your updated velocities
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(L):
        
        ### START CODE HERE ### (approx. 4 lines)
        # compute velocities
        v["dW" + str(l+1)] = beta*v["dW" + str(l+1)] + (1 - beta)*grads['dW' + str(l+1)]
        v["db" + str(l+1)] = beta*v["db" + str(l+1)] + (1 - beta)*grads['db' + str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v["db" + str(l+1)]

    return parameters, v

def initialize_adam(parameters) :
    """
    This functionc is initializes v and s as two python dictionaries with:
    - keys: "dW1", "db1", ..., "dWL", "dbL" 
    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Inputs:
    parameters : python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Outputs: 
    v : python dictionary that will contain the exponentially weighted average of the gradient. Initialized with zeros.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s : python dictionary that will contain the exponentially weighted average of the squared gradient. Initialized with zeros.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    
    for l in range(L):
   
        v["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0], parameters["W" + str(l+1)].shape[1]))
        v["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0], parameters["b" + str(l+1)].shape[1]))
        s["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0], parameters["W" + str(l+1)].shape[1]))
        s["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0], parameters["b" + str(l+1)].shape[1]))
    
    return v, s



def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    This function is update parameters using Adam
    
    Inputs:
    parameters : python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads : python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
                    
    v : Adam variable, moving average of the first gradient, python dictionary
    s : Adam variable, moving average of the squared gradient, python dictionary
    t : Adam variable, counts the number of taken steps
    learning_rate : the learning rate, scalar.
    beta1 : Exponential decay hyperparameter for the first moment estimates 
    beta2 : Exponential decay hyperparameter for the second moment estimates 
    epsilon : hyperparameter preventing division by zero in Adam updates

    Outputs:
    parameters : python dictionary containing your updated parameters 
    v : Adam variable, moving average of the first gradient, python dictionary
    s : Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        
        v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] + (1 - beta1)*grads['dW' + str(l+1)]
        v["db" + str(l+1)] = beta1*v["db" + str(l+1)] + (1 - beta1)*grads['db' + str(l+1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1 - beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1 - beta1**t)

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        
        s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] + (1 - beta2)*np.square(grads['dW' + str(l+1)])
        s["db" + str(l+1)] = beta2*s["db" + str(l+1)] + (1 - beta2)*np.square(grads['db' + str(l+1)])

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1 - beta2**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1 - beta2**t)

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v_corrected["dW" + str(l+1)]/(np.sqrt(s_corrected["dW" + str(l+1)])+epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v_corrected["db" + str(l+1)]/(np.sqrt(s_corrected["db" + str(l+1)])+epsilon)

    return parameters, v, s, v_corrected, s_corrected

if __name__ =='__main__':

    train_X, train_Y = load_dataset()
    # parameters, grads, learning_rate = update_parameters_with_gd_test_case()
    # learning_rate = 0.01
    # parameters = update_parameters_with_gd(parameters, grads, learning_rate)

    # print("W1 =\n" + str(parameters["W1"]))
    # print("b1 =\n" + str(parameters["b1"]))
    # print("W2 =\n" + str(parameters["W2"]))
    # print("b2 =\n" + str(parameters["b2"]))

    # update_parameters_with_gd_test(update_parameters_with_gd)


    # t_X, t_Y, mini_batch_size = random_mini_batches_test_case()
    # mini_batches = random_mini_batches(t_X, t_Y, mini_batch_size)

    # print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    # print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    # print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    # print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    # print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
    # print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    # print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

    # random_mini_batches_test(random_mini_batches)

    # parameters = initialize_velocity_test_case()

    # v = initialize_velocity(parameters)
    # print("v[\"dW1\"] =\n" + str(v["dW1"]))
    # print("v[\"db1\"] =\n" + str(v["db1"]))
    # print("v[\"dW2\"] =\n" + str(v["dW2"]))
    # print("v[\"db2\"] =\n" + str(v["db2"]))

    # initialize_velocity_test(initialize_velocity)


    # parameters, grads, v = update_parameters_with_momentum_test_case()

    # parameters, v = update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
    # print("W1 = \n" + str(parameters["W1"]))
    # print("b1 = \n" + str(parameters["b1"]))
    # print("W2 = \n" + str(parameters["W2"]))
    # print("b2 = \n" + str(parameters["b2"]))
    # print("v[\"dW1\"] = \n" + str(v["dW1"]))
    # print("v[\"db1\"] = \n" + str(v["db1"]))
    # print("v[\"dW2\"] = \n" + str(v["dW2"]))
    # print("v[\"db2\"] = v" + str(v["db2"]))

    # update_parameters_with_momentum_test(update_parameters_with_momentum)

    # parameters = initialize_adam_test_case()

    # v, s = initialize_adam(parameters)
    # print("v[\"dW1\"] = \n" + str(v["dW1"]))
    # print("v[\"db1\"] = \n" + str(v["db1"]))
    # print("v[\"dW2\"] = \n" + str(v["dW2"]))
    # print("v[\"db2\"] = \n" + str(v["db2"]))
    # print("s[\"dW1\"] = \n" + str(s["dW1"]))
    # print("s[\"db1\"] = \n" + str(s["db1"]))
    # print("s[\"dW2\"] = \n" + str(s["dW2"]))
    # print("s[\"db2\"] = \n" + str(s["db2"]))

    # initialize_adam_test(initialize_adam)


    parametersi, grads, vi, si, t, learning_rate, beta1, beta2, epsilon = update_parameters_with_adam_test_case()

    parameters, v, s, vc, sc  = update_parameters_with_adam(parametersi, grads, vi, si, t, learning_rate, beta1, beta2, epsilon)
    print(f"W1 = \n{parameters['W1']}")
    print(f"W2 = \n{parameters['W2']}")
    print(f"b1 = \n{parameters['b1']}")
    print(f"b2 = \n{parameters['b2']}")

    update_parameters_with_adam_test(update_parameters_with_adam)


    

