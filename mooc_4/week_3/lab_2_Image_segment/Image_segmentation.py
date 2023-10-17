import os
import numpy as np 
import pandas as pd 
import imageio
import matplotlib.pyplot as plt
from utils import *

def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')

    return input_image, input_mask


def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block
    
    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns: 
        next_layer, skip_connection --  Next layer and skip connection outputs
    """
    conv = Conv2D(n_filters, # Number of filters
                  3,# Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer= 'he_normal')(inputs)
    conv = Conv2D(n_filters, # Number of filters
                  3,# Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer= 'he_normal')(conv)
    
    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)       
        
    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling is True:
        next_layer = MaxPooling2D(2,strides=2)(conv)        
    else:
        next_layer = conv        
    skip_connection = conv
    
    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block
    
    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """
    
    
    up = Conv2DTranspose(
                 n_filters,    # number of filters
                 3,# Kernel size
                 strides=2,
                 padding='same')(expansive_input)
    
    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=3)
    
    conv = Conv2D(n_filters, # Number of filters
                  3,# Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer= 'he_normal')(merge)
    conv = Conv2D(n_filters, # Number of filters
                  3,# Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer= 'he_normal')(conv)
    return conv


def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    """
    Unet model
    
    Arguments:
        input_size -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns: 
        model -- tf.keras.Model
    """
    inputs = Input(input_size)
    # Contracting Path (encoding)
    # Add a conv_block with the inputs of the unet_ model and n_filters
    
    cblock1 = conv_block(inputs=inputs, n_filters=n_filters*1)
    cblock2 = conv_block(inputs=cblock1[0], n_filters=n_filters*2)
    cblock3 = conv_block(inputs=cblock2[0], n_filters=n_filters*4)
    
    cblock4 = conv_block(inputs=cblock3[0], n_filters=n_filters*8,dropout_prob=0.3)
    cblock5 = conv_block(inputs=cblock4[0], n_filters=n_filters*16,dropout_prob=0.3, max_pooling=False) 
    
    # Expanding Path (decoding)
    # Add the first upsampling_block.
    # From here,at each step, use half the number of filters of the previous block 
    # Use the cblock5[0] as expansive_input and cblock4[1] as contractive_input and n_filters * 8
    
    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters*8)
    # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
    # Note that you must use the second element of the contractive block i.e before the maxpooling layer. 
    
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters*4)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters*2)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters*1)

    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)

    # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
    
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)    
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    """
    Displays the first image of each of the num batches
    """
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = unet.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
             create_mask(unet.predict(sample_image[tf.newaxis, ...]))])


if __name__ == '__main__':
    path = ''
    image_path = os.path.join(path, './data/CameraRGB/')
    mask_path = os.path.join(path, './data/CameraMask/')
    image_list_orig = os.listdir(image_path)
    image_list = [image_path+i for i in image_list_orig]
    mask_list = [mask_path+i for i in image_list_orig]

    #split dataset to unmasked and masked
    image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
    mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)

    for path in zip(image_list_ds.take(3), mask_list_ds.take(3)):
        print(path)

    image_filenames = tf.constant(image_list)
    masks_filenames = tf.constant(mask_list)

    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))

    for image, mask in dataset.take(1):
        print(image)
        print(mask)

    #preprocess dataset
    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(preprocess)
    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(preprocess)
    
    #test encoder block
    input_size=(96, 128, 3)
    n_filters = 32
    inputs = Input(input_size)
    cblock1 = conv_block(inputs, n_filters * 1)
    model1 = tf.keras.Model(inputs=inputs, outputs=cblock1)

    output1 = [['InputLayer', [(None, 96, 128, 3)], 0],
                ['Conv2D', (None, 96, 128, 32), 896, 'same', 'relu', 'HeNormal'],
                ['Conv2D', (None, 96, 128, 32), 9248, 'same', 'relu', 'HeNormal'],
                ['MaxPooling2D', (None, 48, 64, 32), 0, (2, 2)]]

    print('Block 1:')
    for layer in summary(model1):
        print(layer)

    comparator(summary(model1), output1)

    inputs = Input(input_size)
    cblock1 = conv_block(inputs, n_filters * 32, dropout_prob=0.1, max_pooling=True)
    model2 = tf.keras.Model(inputs=inputs, outputs=cblock1)

    output2 = [['InputLayer', [(None, 96, 128, 3)], 0],
                ['Conv2D', (None, 96, 128, 1024), 28672, 'same', 'relu', 'HeNormal'],
                ['Conv2D', (None, 96, 128, 1024), 9438208, 'same', 'relu', 'HeNormal'],
                ['Dropout', (None, 96, 128, 1024), 0, 0.1],
                ['MaxPooling2D', (None, 48, 64, 1024), 0, (2, 2)]]
            
    print('\nBlock 2:')   
    for layer in summary(model2):
        print(layer)
        
    comparator(summary(model2), output2)

    #test decoder block
    input_size1=(12, 16, 256)
    input_size2 = (24, 32, 128)
    n_filters = 32
    expansive_inputs = Input(input_size1)
    contractive_inputs =  Input(input_size2)
    cblock1 = upsampling_block(expansive_inputs, contractive_inputs, n_filters * 1)
    model1 = tf.keras.Model(inputs=[expansive_inputs, contractive_inputs], outputs=cblock1)

    output1 = [['InputLayer', [(None, 12, 16, 256)], 0],
                ['Conv2DTranspose', (None, 24, 32, 32), 73760],
                ['InputLayer', [(None, 24, 32, 128)], 0],
                ['Concatenate', (None, 24, 32, 160), 0],
                ['Conv2D', (None, 24, 32, 32), 46112, 'same', 'relu', 'HeNormal'],
                ['Conv2D', (None, 24, 32, 32), 9248, 'same', 'relu', 'HeNormal']]

    print('Block 1:')
    for layer in summary(model1):
        print(layer)

    comparator(summary(model1), output1)

    
    img_height = 96
    img_width = 128
    num_channels = 3

    unet = unet_model((img_height, img_width, num_channels))
    comparator(summary(unet), outputs.unet_model_output)
    unet.summary()
    unet.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    #display
    for image, mask in image_ds.take(1):
        sample_image, sample_mask = image, mask
        print(mask.shape)
    display([sample_image, sample_mask])

    for image, mask in processed_image_ds.take(1):
        sample_image, sample_mask = image, mask
        print(mask.shape)
    display([sample_image, sample_mask])

    #Training model
    EPOCHS = 5
    VAL_SUBSPLITS = 5
    BUFFER_SIZE = 500
    BATCH_SIZE = 32
    train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    print(processed_image_ds.element_spec)
    model_history = unet.fit(train_dataset, epochs=EPOCHS)

    #plot acc
    plt.plot(model_history.history["accuracy"])

    #show prediction
    show_predictions(train_dataset, 6)
