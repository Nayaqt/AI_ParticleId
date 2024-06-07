from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D


import logging

def model_binary_classifier(config = None, input_shape = None):
    """
    Creates a binary classification model based on a configuration file

    Parameters : 
    - config (str) : Path to configuration file
    - input_shape (tuple) : Shape of the training data

    Returns : 
    - binary_classifier (keras model) : Model which architecture optimized for binary classification

    """

    #Load NN structure from configuration file 
    NN_structure = config['NN_structure']

    #Create the model from config_file
    input = Input(shape=(input_shape, ))
    x = Dense(NN_structure['input_layer'], activation = 'relu')(input)

    for layer in NN_structure['dense_layers']:
        x = Dense(layer, activation = 'relu')(x)

    output = Dense(1, activation = 'sigmoid')(x)

    binary_classifier = Model(inputs=input, outputs = output)

    #Print summary of the model to terminal
    binary_classifier.summary()

    #Compile the model
    model_optimizer = Adam(learning_rate = NN_structure['lr'])
    binary_classifier.compile(
        loss="bce",
        optimizer=model_optimizer,
        metrics=["accuracy"],
    )

    return binary_classifier

def model_multi_classifier(config = None, input_shape = None):
    """
    Creates a multi-classification model based on a configuration file

    Parameters : 
    - config (str) : Path to configuration file
    - input_shape (tuple) : Shape of the training data

    Returns : 
    - multi_classifier (keras model) : Model which architecture optimized for binary classification

    """

    #Load NN structure from configuration file 
    NN_structure = config['NN_structure']

    #Load NN parameters 
    batch_norm = NN_structure['batch_normalization']
    dropout = NN_structure['dropout']

    # Define the architecture depending on whether to use dropout or batch_norm : 
    input = Input(shape=(input_shape, )) #Input layer with specified shape

    x = Dense(NN_structure['input_layer'], activation = 'relu')(input)  #First dense layer

    for i, layer in enumerate(NN_structure['dense_layers']):
        x = Dense(layer, activation = 'relu')(x)    #Define multiple dense layers

        if dropout != 0:
            x = Dropout(rate = dropout)(x)  #Apply dropout if not zero
        
        if batch_norm :
            x = BatchNormalization()(x) #Apply batchNorm if True

    output = Dense(3, activation = 'softmax')(x)    #Output layer with softmax activation

    multi_classifier = Model(inputs=input, outputs = output)    #Define model

    #Print summary of the model to terminal
    multi_classifier.summary()

    #Define model opt
    model_optimizer = Adam(learning_rate = NN_structure['lr'])

    #Compile the model
    multi_classifier.compile(
        loss="categorical_crossentropy",
        optimizer=model_optimizer,
        metrics=["accuracy"],
    )

    return multi_classifier


def classifier(X_train, y_train, W_train, X_val, y_val, config_file, results_directory, model_type):
    """
    Fits (trains) a model depending on its type (binary, multi-class)

    Parameters : 
    - X_train (pd.df) : Training data
    - y_train (pd.df) : Labels for training data
    - W_train (pd.df) : Weights to be used during training
    - X_val (pd.df) : Validation data
    - y_val (pd.df) : Validation labels
    - config_file (yaml file) : Configuration file
    - results_directory (str) : Path to results directory
    - model_type (str) : binary or multi-class

    """
    if model_type == 'binary':
        my_model = model_binary_classifier(config_file, X_train.shape[1])
    elif model_type == 'multi_class':
        my_model = model_multi_classifier(config_file, X_train.shape[1])
    else :
        print('Unknown model_type')

    # Load parameters from config_file
    NN_structure = config_file['NN_structure']
    nEpochs = NN_structure['nEpochs']

    # Define callbacks :

    #ModelCheckpoint saves the model to a directory on each epoch.
    model_chkpt = ModelCheckpoint(
        f"{results_directory}" + "/model_{epoch:02d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=True,
        validation_batch_size=NN_structure["batch_size"],
        save_weights_only=False,
    )   
    
    #EarlyStopping stops the training when the given metric stop improving after a given number of epochs (patience)
    early_stop = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience = 10
    )
    
    #ReduceLROnPlateau reduces the learning rate if after (patience) epochs, the given metric (monitor) has not improved.
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.2,
        patience=3,
        mode='auto',
        cooldown=3,
        min_lr=0.000001,
    )

    # Fit the model and return the model for evaluation.
    logging.info("Training started")
    return my_model, my_model.fit(
        X_train,
        y_train,
        sample_weight = W_train,
        epochs=nEpochs,
        validation_data= (X_val, y_val),
        callbacks = [reduce_lr], #Can include other callbacks, such as ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
        steps_per_epoch=len(X_train) / NN_structure["batch_size"],
        use_multiprocessing = True,
        verbose = 1
    )
    
def make_multi_cnn(config_file):
    """
    Creates a CNN model based on a configuration file 

    Parameter :
    - config_file (yaml) : Configuration file containing the model hyperparameters (learning rate etc..)

    """

    #Load NN parameters from configuration file:
    NN_structure = config_file['NN_structure']
    lr = NN_structure['lr']
    nEpochs = NN_structure['nEpochs']
    batch_size = NN_structure['batch_size']
    nFilters = NN_structure['nFilters']
    k = NN_structure['kernel_size']

    #Make the model architecture
    ins = Input(shape=(19,19,1))    #Input layer for 19x19 grayscale images
    h1 = Conv2D(filters = nFilters,
                kernel_size = (k,k),
                data_format = 'channels_last',
                activation = 'relu'
            )(ins)      #First convolutional layer
    h2 = Conv2D(filters = nFilters,
                kernel_size = (k,k),
                data_format = 'channels_last',
                activation = 'relu'
                )(h1)    #Second convolutional layer
    h3 = MaxPooling2D(pool_size=(2,2))(h2)      #First max pooling layer
    h4 = Conv2D(filters = 2*nFilters,
                kernel_size = (k,k),
                data_format = 'channels_last',
                activation = 'relu',
                padding = 'same'
            )(h3)       # Third convolutional layer with doubled filters
    h5 = Conv2D(filters = 2*nFilters,
                kernel_size = (k,k),
                data_format = 'channels_last',
                activation = 'relu',
                padding = 'same'
                )(h4)    # Fourth convolutional layer
    h6 = MaxPooling2D(pool_size=(2,2))(h5)  # Second max pooling layer

    h7 = Flatten()(h6)  # Flatten the 3D output to 1D

    #Using 'relu' as activation :
    h10 = Dense(512, activation ='relu')(h7) #First fully connected layer
    h11 = Dense(256, activation = 'relu')(h10) #Second fully connected layer
    h12 = Dense(64, activation = 'relu')(h11) #Third fully connected layer
    h13 = Dense(32, activation = 'relu')(h12) #Fourth fully connected layer
    outs = Dense(3, activation = 'softmax')(h13) # Output layer with softmax activation for 3 classes

    model = Model(inputs = ins, outputs = outs) #Define model

    #Print a summary of the model
    model.summary()

    model_optimizer = Adam(learning_rate = lr)
    model.compile(loss= 'categorical_crossentropy', optimizer = model_optimizer, metrics=['accuracy'])

    return model

def make_binary_cnn(config_file):
    """
    Creates a binary CNN model based on a configuration file 

    Parameter :
    - config_file (yaml) : Configuration file containing the model hyperparameters (learning rate etc..)    
    """

    #Load NN parameters from configuration file:
    NN_structure = config_file['NN_structure']
    lr = NN_structure['lr']
    nFilters = NN_structure['nFilters']
    k = NN_structure['kernel_size']

    #Define the model architecture
    ins = Input(shape=(19,19,1))    #Input layer for 19x19 grayscale images
    h1 = Conv2D(filters = nFilters,
                kernel_size = (k,k),
                data_format = 'channels_last',
                activation = 'relu'
            )(ins)  #First convolutional layer
    h2 = Conv2D(filters = nFilters,
                kernel_size = (k,k),
                data_format = 'channels_last',
                activation = 'relu'
                )(h1)    #Second convolutional layer
    h3 = MaxPooling2D(pool_size=(2,2))(h2)  #First MaxPooling layer
    h4 = Conv2D(filters = 2*nFilters,
                kernel_size = (k,k),
                data_format = 'channels_last',
                activation = 'relu',
                padding = 'same'
            )(h3)   #Third convolutional layer
    h5 = Conv2D(filters = 2*nFilters,
                kernel_size = (k,k),
                data_format = 'channels_last',
                activation = 'relu',
                padding = 'same'
                )(h4)    #Fourth convolutional layer, with doubled filters
    h6 = MaxPooling2D(pool_size=(2,2))(h5)  #Second MaxPooling layer

    h7 = Flatten()(h6)  #Flatten the output to 1D

    h10 = Dense(256, activation ='relu')(h7)    #First fully-connected layer
    h11 = Dense(128, activation = 'relu')(h10)  #Second fully-connected layer
    h12 = Dense(64, activation = 'relu')(h11)   #Third fully-connected layer
    h13 = Dense(32, activation = 'relu')(h12)   #Fourth fully-connected layer
    outs = Dense(1, activation = 'sigmoid')(h13)    #Output layer, with sigmoid as activation

    model = Model(inputs = ins, outputs = outs) #define model

    #Print a summary of the model
    model.summary()

    model_optimizer = Adam(learning_rate = lr)
    model.compile(loss= 'bce', optimizer = model_optimizer, metrics=['accuracy'])

    return model

def make_cnn(X_train, y_train, X_val, y_val, config_file, model_type, results_directory):
    """
    Fits (trains) a CNN model depending if its for binary or multi-classification

    Parameters : 
    - X_train (pd.df) : Training data
    - y_train (pd.df) : Labels for training data
    - X_val (pd.df) : Validation data
    - y_val (pd.df) : Validation labels
    - config_file (yaml file) : Configuration file
    - model_type (str) : binary or multi-class
    - results_directory (str) : Path to results directory
    
    """
    if model_type == 'binary':
        my_model = make_binary_cnn(config_file)
    elif model_type == 'multi_class':
        my_model = make_multi_cnn(config_file)
    else :
        print('Unknown model_type')
    
    #Load NN parameters from configuration file:
    NN_structure = config_file['NN_structure']
    nEpochs = NN_structure['nEpochs']
    batch_size = NN_structure['batch_size']

    #ReduceLROnPlateau reduces the learning rate if after (patience) epoch the given metric (monitor) has not improved.
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.2,
        patience=3,
        mode='auto',
        cooldown=3,
        min_lr=0.000001,
    )

    #Saves the model to directory
    model_chkpt = ModelCheckpoint(
        f"{results_directory}" + "/model_{epoch:02d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=True,
        validation_batch_size=NN_structure["batch_size"],
        save_weights_only=False,
    )

    logging.info(f'Training started!')
    return my_model, my_model.fit(
        X_train,
        y_train,
        batch_size = batch_size,
        epochs = nEpochs,
        validation_data = (X_val, y_val),
        callbacks = [reduce_lr],
        )
    