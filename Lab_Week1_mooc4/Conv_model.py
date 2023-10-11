import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

np.random.seed(1)

def import_happy_dataset():
    #Load the HAPPY Data and Split the Data into Train/Test Sets then normalize and reshape sets
    X_train_orig_happy, Y_train_orig_happy, X_test_orig_happy, Y_test_orig_happy, classes = load_happy_dataset()
    # Normalize image vectors
    X_train_happy = X_train_orig_happy/255.
    X_test_happy = X_test_orig_happy/255.
    # Reshape
    Y_train_happy = Y_train_orig_happy.T
    Y_test_happy = Y_test_orig_happy.T

    print ("number of training examples = " + str(X_train_happy.shape[0]))
    print ("number of test examples = " + str(X_test_happy.shape[0]))
    print ("X_train shape: " + str(X_train_happy.shape))
    print ("Y_train shape: " + str(Y_train_happy.shape))
    print ("X_test shape: " + str(X_test_happy.shape))
    print ("Y_test shape: " + str(Y_test_happy.shape))

    return X_train_happy, Y_train_happy, X_test_happy, Y_test_happy

#FUNCTION: happyModel

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential([
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            
            ## Conv2D with 32 7x7 filters and stride of 1
            
            ## BatchNormalization for axis 3
            
            ## ReLU
            
            ## Max Pooling 2D with default parameters
            
            ## Flatten layer
            
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            tfl.ZeroPadding2D(padding=(3, 3),input_shape=(64,64,3)),
        
            ## Conv2D with 32 7x7 filters and stride of 1
            tfl.Conv2D(filters=32, kernel_size=7, strides=(1,1), input_shape=[64, 64, 3]),
        
            ## BatchNormalization for axis 3
            tfl.BatchNormalization(axis=3),    
            
            ## ReLU
            tfl.ReLU(max_value=None, negative_slope=0.0, threshold=0.0),

            ## Max Pooling 2D with default parameters
            tfl.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),

            ## Flatten layer
            tfl.Flatten(),
        
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            tfl.Dense(1,activation="sigmoid")
        ])
    
    return model

def import_sign_dataset():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))

    return X_train, Y_train, X_test, Y_test
    # GRADED FUNCTION: convolutional_model

def signModel(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    # Z1 = None
    ## RELU
    # A1 = None
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    # P1 = None
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    # Z2 = None
    ## RELU
    # A2 = None
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    # P2 = None
    ## FLATTEN
    # F = None
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    # outputs = None
    # YOUR CODE STARTS HERE
    
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tf.keras.layers.Conv2D(filters=8, kernel_size=4, padding="same")(input_img)
    ## RELU
    A1 = tf.keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.keras.layers.MaxPool2D(pool_size=(8,8), strides=(8,8), padding='same')(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tf.keras.layers.Conv2D(filters=16, kernel_size=2,strides=(1,1), padding="same")(P1)
    ## RELU
    A2 = tf.keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.keras.layers.MaxPool2D(pool_size=(4,4), strides=(4,4), padding='same')(A2)
    ## FLATTEN
    F = tf.keras.layers.Flatten()(P2)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs = tf.keras.layers.Dense(6,activation="softmax")(F)
    
    # YOUR CODE ENDS HERE
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model

def menu(options):
    if options == 1:
        X_train_happy, Y_train_happy, X_test_happy, Y_test_happy = import_happy_dataset()
        happy_model = happyModel()
        happy_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        happy_model.summary()

        # Fit the parameter
        happy_model.fit(X_train_happy, Y_train_happy, epochs=10, batch_size=16)
        print("Evaluate model:")
        happy_model.evaluate(X_test_happy, Y_test_happy)
    if options == 2:
        X_train, Y_train, X_test, Y_test = import_sign_dataset()
        sign_model = signModel((64, 64, 3))
        sign_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        sign_model.summary()
        # Fit the parameter
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
        sign_model.fit(train_dataset, epochs=100, validation_data=test_dataset)
        sign_model.evaluate(X_test_happy, Y_test_happy)


def display():
    """
    Display the menu option
    """

    menu_dict = {1:'CNN model with happy dataset',
                 2:'CNN model with sign dataset',
                 3:'Exit'}
    print('------Menu------')
    for key, val in menu_dict.items():
        print(f'{key} : {val}')
    print('----------------')

if __name__ == '__main__':
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
                break