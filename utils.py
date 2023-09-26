import numpy as np
from lr_utils import load_dataset
import copy


def sigmoid(z):
    s = np.divide(1, 1 + np.exp(-z))
    return s

def initialize_parameters(dim):
    w= np.zeros(shape=(dim,1))
    b= float(0.0)
    return w, b

def propagate(w,b , X, Y):
    m = X.shape[1]
    y = np.dot(w.T, X) + b
    A= sigmoid(y)
    cost = (-1/ m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = (1 / m) * np.dot(X , (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    cost = np.squeeze(np.array(cost))   
    grads = {"dw": dw,
             "db": db}
    return grads, cost

def optimization(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    for i in range(num_iterations):
        costs =[]
        grads, cost = propagate(w,b,X,Y)
        dw = grads['dw']
        db= grads['db']
        w = w - learning_rate*dw
        b= b - learning_rate* db
        if i % 100 == 0:
                costs.append(cost)        
                if print_cost:
                    print ("Cost after iteration %i: %f" %(i, cost))
        
        params = {"w": w,
                "b": b}
        
        grads = {"dw": dw,
                "db": db}
    
    return params, grads, costs

def predict(w: np.array,b,X) -> np.array:
    """
    
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    return Y_prediction

