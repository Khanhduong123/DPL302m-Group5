from ultil_1 import *
import numpy as np
import matplotlib.pyplot as plt


def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    Design a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Inputs:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent 
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    
    Outputs:
    parameters -- parameters learnt by the model
    """
        
    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]
    
    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)

    for i in range(num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)
        
        # Loss
        cost = compute_loss(a3, Y)

        # Backward propagation.
        grads = backward_propagation(X, Y, cache)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def menu(options):
    if options == 1:
        #Initialization zeros parameters
        parameters = initialize_parameters_zeros([3, 2, 1])
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))
        initialize_parameters_zeros_test(initialize_parameters_zeros)

        #train and predict result
        parameters = model(train_X, train_Y, initialization = "zeros")
        print ("On the train set:")
        predictions_train = predict(train_X, train_Y, parameters)
        print ("On the test set:")
        predictions_test = predict(test_X, test_Y, parameters)
    
    
    elif options ==2 :
        # Initialization random parameters
        parameters = initialize_parameters_random([3, 2, 1])
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))
        initialize_parameters_random_test(initialize_parameters_random)

        #train and predict result
        parameters = model(train_X, train_Y, initialization = "random")
        print ("On the train set:")
        predictions_train = predict(train_X, train_Y, parameters)
        print ("On the test set:")
        predictions_test = predict(test_X, test_Y, parameters)

        #Plot the decision boundary
        plt.title("Model with large random initialization")
        axes = plt.gca()
        axes.set_xlim([-1.5,1.5])
        axes.set_ylim([-1.5,1.5])
        plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    
    
    elif options == 3:
        #Initialize the Xavier parameters
        parameters = initialize_parameters_he([2, 4, 1])
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))
        initialize_parameters_he_test(initialize_parameters_he)

        #train and predict result
        parameters = model(train_X, train_Y, initialization = "he")
        print ("On the train set:")
        predictions_train = predict(train_X, train_Y, parameters)
        print ("On the test set:")
        predictions_test = predict(test_X, test_Y, parameters)


def display():
    """
    Display the menu option
    """

    menu_dict = {1:'A model with zero parameters',
                 2:'A model with random initialization',
                 3:'A model with Xavier initialization',
                 4:'Exits'}
    print('------Menu------')
    for key, val in menu_dict.items():
        print(f'{key} : {val}')
    print('----------------')

if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_dataset()
    while True:
        display()
        menu_options= int(input('Enter your choose: '))
        if menu_options > 4:
            print('Choose input in range 1 to 4')
        else:
            if menu_options == 1:
                menu(1)
            elif menu_options == 2:
                menu(2)
            elif menu_options == 3: 
                menu(3)
            elif menu_options == 4:
                break
