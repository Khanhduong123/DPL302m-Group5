from utils import *


def load_data(directory,BATCH_SIZE,IMG_SIZE ):
    train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)
    
    validation_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='validation',
                                             seed=42)
    return train_dataset , validation_dataset


def data_augmenter():
    '''
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    '''
    ### START CODE HERE
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip('horizontal')) #
    data_augmentation.add(RandomRotation(0.2))
    ### END CODE HERE
    
    return data_augmentation

def plot_data_augmenter(trainset):
    for image, _ in trainset.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = augmenter(tf.expand_dims(first_image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')


def alpaca_model(image_shape=None, data_augmentation=data_augmenter()):
    ''' Define a tf.keras model for binary classification out of the MobileNetV2 model
    Arguments:
        image_shape -- Image width and height
        data_augmentation -- data augmentation function
    Returns:
    Returns:
        tf.keras.model
    '''
    
    
    input_shape = image_shape + (3,) #RGB channel

    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False, # <== Important!!!!
                                                   weights='imagenet') # From imageNet
    
    # freeze the base model by making it non trainable
    base_model.trainable = False 

    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=input_shape) 
    x = data_augmentation(inputs)
    x = preprocess_input(x) 
    x = base_model(x, training=False) #pass preprocess input thourgh input model
    x = tfl.GlobalAveragePooling2D()(x) 
    # include dropout with probability of 0.2 to avoid overfitting
    x = tfl.Dropout(0.2)(x)

    outputs = tfl.Dense(1,'linear')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

def fine_tunning():
    base_model = model2.layers[4] #take layer 4 in previous model
    base_model.trainable = True #set up this layer for trainning
    # see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 120 #fine-tune in 120th layer, layer before nontrain

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # Define a BinaryCrossentropy loss function.
    loss_function=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # Define an Adam optimizer with a learning rate of 0.1 * base_learning_rate
    optimizer = tf.keras.optimizers.Adam(lr=base_learning_rate/10)
    # Use accuracy as evaluation metric
    metrics=['accuracy']

    model2.compile(loss=loss_function,
                optimizer = optimizer,
                metrics=metrics)


if __name__ =='__main__':
    BATCH_SIZE = 32
    IMG_SIZE = (160, 160)
    directory = "dataset/"
    train_dataset , validation_dataset = load_data(directory,BATCH_SIZE,IMG_SIZE)
    augmenter = data_augmenter()
    plot_data_augmenter(train_dataset)
    AUTOTUNE = tf.data.experimental.AUTOTUNE # Tối ưu hoá đường dẫn dữ liệu, tập dữ liệu được tìm và nạp trc để cải thiện tốc độ truyền tải dữ liệu
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    #download model pre-train
    IMG_SHAPE = IMG_SIZE + (3,) #RGB channel
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=True,
                                               weights='imagenet')
    base_model.summary()


    nb_layers = len(base_model.layers)
    print(base_model.layers[nb_layers - 2].name)
    print(base_model.layers[nb_layers - 1].name)
    

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    #decode and predict using mobilenet
    base_model.trainable = False
    image_var = tf.Variable(preprocess_input(image_batch))
    pred = base_model(image_var)
    tf.keras.applications.mobilenet_v2.decode_predictions(pred.numpy(), top=2)

    #trainfer learning
    model2 = alpaca_model(IMG_SIZE, augmenter)
    model2.summary()
    base_learning_rate = 0.001
    model2.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    history = model2.fit(train_dataset, validation_data=validation_dataset, epochs=5)
    acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']

    #fine -tunning
    fine_tunning()
    fine_tune_epochs = 5
    total_epochs =  5 + fine_tune_epochs

    history_fine = model2.fit(train_dataset,
                            epochs=total_epochs,
                            initial_epoch=history.epoch[-1],
                            validation_data=validation_dataset)



    # alpaca_summary = [['InputLayer', [(None, 160, 160, 3)], 0],
    #                 ['Sequential', (None, 160, 160, 3), 0],
    #                 ['TensorFlowOpLayer', [(None, 160, 160, 3)], 0],
    #                 ['TensorFlowOpLayer', [(None, 160, 160, 3)], 0],
    #                 ['Functional', (None, 5, 5, 1280), 2257984],
    #                 ['GlobalAveragePooling2D', (None, 1280), 0],
    #                 ['Dropout', (None, 1280), 0, 0.2],
    #                 ['Dense', (None, 1), 1281, 'linear']] #linear is the default activation

    # # comparator(summary(model2), alpaca_summary)

    # # for layer in summary(model2):
    # #     print(layer)


    assert(augmenter.layers[0].name.startswith('random_flip')), "First layer must be RandomFlip"
    assert augmenter.layers[0].mode == 'horizontal', "RadomFlip parameter must be horizontal"
    assert(augmenter.layers[1].name.startswith('random_rotation')), "Second layer must be RandomRotation"
    assert augmenter.layers[1].factor == 0.2, "Rotation factor must be 0.2"
    assert len(augmenter.layers) == 2, "The model must have only 2 layers"

    print('\033[92mAll tests passed!')