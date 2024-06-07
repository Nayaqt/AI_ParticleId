#Imports

import yaml
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def read_images_hdf(data_directory, nJets):
    """
    Loads image data from h5 file and returns the data and labels as separate.

    Parameters :
    - data_directory (str) : path to directory containing the image data.
    - nJets (int) : number of jets (images) to read. 

    Returns : 
    - X_data (numpy histogram) : Image data stored as a numpy histogram
    - y_data (np.array) : Labels corresponding to each jet type
    """
    hf = h5py.File(f'{data_directory}_{nJets}.h5', 'r')
    X_data = hf['X_data'][:]
    y_data = hf['y_data'][:]
    hf.close()
    return(X_data, y_data)

def save_dataset(X_data, y_data, saving_directory, nJets):
    """
    Create a h5py file in which images and labels data is saved.
    
    Parameters :
    - X_data (numpy histogram) : Image data stored as a numpy histogram
    - y_data (np.array) : Labels corresponding to each jet type
    - saving_directory (str) : Name of the directory in which to save the data
    - nJets (int) : number of jets (used for naming the directory)
    """
    
    h5f = h5py.File(f'{saving_directory}_{nJets}.h5', 'w')
    h5f.create_dataset('X_data', data = X_data)
    h5f.create_dataset('y_data', data = y_data)
    h5f.close()

def prepare_dataset(data_dir, jets, nJets):
    """
    Merges the datasets from the specified jets and creates a column with the corresponding label.

    Parameters :

    - data_dir (str) : Path to directory containing the data
    - jets (list) : List of jets to load (qcd, top or wz)
    - nJets (int) : [optional] Number of jets to load 

    Returns:

    - dataframe (pandas dataframe) : Dataframe containg jet data with the last column corresponding to the label of the jet.
    """    
    X = []
    for jet in jets :
        full_dataset = pd.read_hdf(f"{data_dir}/{jet}_small.h5")
        # Incude number of specified jets,if not specified, include all jets.
        if nJets is not None:
            dataset = full_dataset[:nJets]
            dataset['label'] = jet                  
            X.append(dataset)
        if nJets is None :
            dataset= full_dataset[:]
            dataset['label'] = jet
            X.append(dataset)
    dataframe = pd.concat(X, ignore_index=True)

    return dataframe

def normalize_get_weights(data, jet_types):
    """
    Normalizes the dataset using scikit.preprocessing.StandardScaler
    Returns 2 dataframes containing X & y (variables & labels) as well as the weights and a dict containing class weights.

    Parameters :
    - data (pd.df) : Dataframe containing X data and labels
    - jet_types (list) : List containing the jet_types considered

    Returns :
    - X (pd.df) : Normalized dataframe containing only X data (without labels)
    - y (pd.series) : Labels
    - class_weights (dict) : Dictionnary containing the weights associated to each jet type ('qcd': 0.05) (used to one-hot-encode labels)
    - weights (np.array) : Array containing the weights for each jet type.
    """
    y = data['label']
    X = data.drop(columns=['label'])

    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                  classes = np.unique(y),
                                                  y = y)
    class_weights = dict(zip(np.unique(y), class_weights))

    weights = np.zeros(len(X)) 
    for jet in jet_types:
        weights[np.where(y == jet)] = class_weights[jet]
    
    return X, y, class_weights, weights

def get_mass_weights(X, y, weights, jet_types):
    """
    Compute the mass weights in order to reduce mass dependency in predictions.

    Parameters :
    - X (pd.df) : Normalized dataframe containing only X data (without labels)
    - y (pd.list) : Labels
    - weights (np.array) : Array containing the weights for each jet type.

    Returns : 
    - mass_weights (np.array) : Array containing the mass weights
    """
    values = []
    for jet in jet_types: 

        value, bins = np.histogram(X[y==jet, 3], bins = 50, weights = weights[y==jet])
        values.append(value)

    # Compute bin weights by normalizing the data:
    binweights = np.nan_to_num(values[0]/values[1:], nan=0, posinf=+1, neginf=-1)

    w_mass = np.ones_like(weights)

    for j, jet in enumerate(jet_types[1:]):
        for i in range(len(bins)-1):
            w_mass[(X[:,3] >= bins[i]) & (X[:,3] < bins[i+1]) & (y == jet)] *= binweights[j][i]
    
    # Compute mass weights:
    mass_weights = weights * w_mass

    return mass_weights


def load_yaml(file: str= None):
    """
    Loads a yaml configuration file
    """
    try :
        with open(file, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError('File could not be found.')
    

def encoder(y, class_weights, jet_types):
    """
    Encodes the target depending on the number of target class, using LabelEncoder (maps target to 0s and 1s) or OneHotEncoder (returns columns of 0s and 1s, useful pour >2 classification)

    Parameters : 
    - y (pd.series) : Column containing the labels
    - class_weights (np.array) : weights associated to each class.
    - jet_types (list) : List of jet types considered (qcd, top, wz)

    Returns : 
    y (pd.df) : Encoded labels
    class_weights (np.array) : Array containing only weights now.
    """
    if len(np.unique(y)) > 2 :
        onehotencoder = OneHotEncoder()
        y = y.values.reshape(-1,1)
        y = onehotencoder.fit_transform(y).toarray()        
    else : 
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)
    
    for i, jet in enumerate(jet_types):
        class_weights[i] = class_weights.pop(jet)
    
    return y, class_weights

def split_data(X, y, weights):
    """
    Splits the data into train/val/test sets using a random seed (42)

    Parameters : 
    - X (pd.df) : Dataframe containing X data
    - y (pd.df) : Dataframe containing y data (labels)
    - weights (list) : List of weights to be used for training and testing

    Returns : 
    - X_train (pd.df) : Dataset containing the data used for training.
    - X_test (pd.df) : Dataset containing the data used for testing
    - X_val (pd.df) : Dataset containing the data used for validation.
    - y_train (pd.df) : Dataset containing the labels used for training.
    - y_test (pd.df) : Dataset containing the labels used for testing
    - y_val (pd.df) : Dataset containing the labels used for validation.
    - W_train (pd.series) : List of weights used for training

    """            

    #Split data into 2 sets sets:
    X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(X, y, weights, test_size = 0.2, random_state = 42)
    #Further split the training set into a training and validation set:
    X_train, X_val, y_train, y_val, W_train, W_val = train_test_split(X_train, y_train, W_train, test_size = 0.25, random_state = 42)   

    return X_train, X_test, X_val, y_train, y_test, y_val, W_train

def split_cnn_data(X, y):
    """
    Splits the data into train/val/test sets using a random seed (42), without using weights (no need for image data)

    Parameters : 
    - X (pd.df) : Dataframe containing X data
    - y (pd.df) : Dataframe containing y data (labels)

    Returns : 
    - X_train (pd.df) : Dataset containing the data used for training.
    - X_test (pd.df) : Dataset containing the data used for testing
    - X_val (pd.df) : Dataset containing the data used for validation.
    - y_train (pd.df) : Dataset containing the labels used for training.
    - y_test (pd.df) : Dataset containing the labels used for testing
    - y_val (pd.df) : Dataset containing the labels used for validation.
    """                      

    #Split data into 2 sets sets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    #Further split the training set into a training and validation set:
    X_train, X_val, y_train, y_val,  = train_test_split(X_train, y_train,  test_size = 0.25, random_state = 42)   

    return X_train, X_test, X_val, y_train, y_test, y_val

def plot_epoch_performance(h, result_dir):
    """
    Creates 2 plots : 
    - Training & validation loss per epoch
    - Accuracy per epoch

    Parameters : 
    - h (callbacks.method) : History of the fitted model 
    - results dir (str) : Path to results directory

    Returns :
    - None, but plots are saved
    """

    #Shift validation set epochs to account loss/accuracy is computed at the end of an epoch.
    epochs = np.arange(0, len(h.history['loss']), 1) 

    #Creates plot folder in case it does not exist
    os.makedirs(f"{result_dir}/plots" ,exist_ok=True)

    #Plot loss from history
    plt.plot(epochs - 0.5,h.history['loss'])
    plt.plot(epochs, h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f"{result_dir}/plots/loss_evolution.png")
    plt.close()


    #Plot accuracy from history
    plt.plot(epochs - 0.5, h.history['accuracy'])
    plt.plot(epochs, h.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f"{result_dir}/plots/accurac_evolution.png")
    plt.close()


def make_images(data_dir, jet, nJets):
    """
    Creates 2d histogram in the eta/phi plane with energy as 'height' to be used as images in a CNN.

    Parameters : 
    - data_dir (str) : Path to directory containing the raw image data
    - jet (str) : Jet type to consider
    - nJets (int) : Number of images to create

    """

    jet_images = []

    #Load data
    small_data = pd.read_hdf(f"{data_dir}/{jet}_small.h5") 
    const_file = h5py.File(f"{data_dir}/{jet}_const.h5")
    
    const_data = const_file['constituents'][:nJets]

    #Extract eta and phi
    eta = small_data.eta[:nJets, np.newaxis] - const_data[:nJets,:,1]
    phi = small_data.phi[:nJets, np.newaxis] - const_data[:nJets,:,2]

    # Account for phi bounds
    phi = np.where(phi < -np.pi, phi+2*np.pi, phi)
    phi = np.where(phi > +np.pi, phi-2*np.pi, phi)

    # Compute energy (relativistic definition)
    fracE = const_data[:,:,0]*np.cosh(const_data[:,:,1])/np.sum(const_data[:,:,0]*np.cosh(const_data[:,:,1]), axis=1)[::,np.newaxis]

    #Create 2d hist :
    for i in range(len(fracE)):
        hist_2D, _ , _ = np.histogram2d(eta[i].ravel(), phi[i].ravel(), weights = (fracE[i].ravel())/(len(fracE)), bins=(np.arange(-0.975,0.975,0.1),np.arange(-0.975,0.975,0.1)))
        jet_images.append(hist_2D.T)
    
    return jet_images

def make_dataset_images(config_file, nJets, jets):

    """
    Creates two datasets from images histoframs, one containing the histograms, the other with corresponding labels.

    Parameters : 
    - config_file (str) : Path to configuration file
    - nJets (int) : Number of images to create
    - jets (list) : List of jets considered
    """

    #Get images dataset depending on the selected jets:
    if len(jets) == 2:
        if 'qcd' and 'wz' in jets:
            qcd_images = make_images(config_file['data_directory'], 'qcd', nJets)
            wz_images = make_images(config_file['data_directory'], 'wz', nJets)

            X_dataset = np.concatenate((qcd_images, wz_images))
            y_dataset = np.array([0]*len(qcd_images) + [1]*len(wz_images))

        if 'qcd' and 'top' in jets:
            qcd_images = make_images(config_file['data_directory'], 'qcd', nJets)
            top_images = make_images(config_file['data_directory'], 'top', nJets)

            X_dataset = np.concatenate((qcd_images, top_images))
            y_dataset = np.array([0]*len(qcd_images) + [1]*len(top_images))
    
    if len(jets) == 3:
        if 'qcd' and 'wz' and 'top' in jets:
            qcd_images = make_images(config_file['data_directory'], 'qcd', nJets)
            wz_images = make_images(config_file['data_directory'], 'wz', nJets)
            top_images = make_images(config_file['data_directory'], 'top', nJets)

            X_dataset = np.concatenate((qcd_images, wz_images, top_images))
            y_dataset = np.array([0]*len(qcd_images) + [1]*len(wz_images) + [2]*len(top_images))  

    return X_dataset, y_dataset

def normalize(X_data, multi=255):
    """
    Normalize images in [0,multi] range.

    Parameters : 
    - X_data (np.hist2d): Dataset containing images

    Returns : 
    - Normalized images
    """
    return (X_data/np.max(X_data)*multi).astype(int)

def normalize_reshape(X_data):
    """
    Normalizes images using normalize function and reshapes it to match what is expected from TensorFlow

    Parameters : 
    - X_data (pd.df) : dataframe containing images

    Returns : 
    - X (pd.df) : Rescaled and reshaped data
    """

    #Normalize all images using a map
    normalized_map = map(normalize, X_data)
    X_data = np.array(list(normalized_map))

    #Reshaping
    X = np.stack(X_data)
    X = X.reshape(X.shape + (1,)).astype('float32')
    X /= 255

    return X
    
def cnn_encoder(y):
    """
    Encodes the target without using class_weights (in case only a CNN is trained). OneHotEncoder it used if N_classes >2.
    
    Parameters :
    - y (pd.series) : Series containing the labels

    Returns : 
    - y (pd.series or pd.df) : Encoded target (depending on the number of target, either a df or a series)

    """
    if len(np.unique(y)) > 2 :
        onehotencoder = OneHotEncoder()
        y = y.reshape(-1,1)
        y = onehotencoder.fit_transform(y).toarray()        
    else : 
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)    
    
    return y

def make_images_small_dataset(X_test_images, X_test_small, y_test_small, config_file):

    """
    Creates a dataframe which contains small.h5 data (detector parameters) as well as CNN predictions as VARIABLES, with labels associated labels.

    Parameters :
    - X_test_images (pd.df) : Dataframe used for testing (used to get CNN predictions)
    - X_test_small (pd.df) : Dataframe containing the data from the detector
    - y_test_small (pd.df) : labels corresponding to X_test_small
    - config_file (str) : Path to the configuration file

    Returns :
    - X (pd.df) : Dataframe containing data from the detector, as well as predictions from the CNN as new inputs, and the corresponding labels.
    """

    #Load both datasets
    cnn_model = load_model(config_file['cnn_model'])
    cnn_predictions = cnn_model.predict(X_test_images)

    # Create lists of predictions for each class
    CNNqcd = []
    CNNwz = []
    CNNtop = []

    for i in range(len(cnn_predictions)):
        CNNqcd.append(cnn_predictions[i][0])
        CNNwz.append(cnn_predictions[i][1])
        CNNtop.append(cnn_predictions[i][2])

    #Create a dataset using small.h5 variables as well as predictions :
    X = pd.DataFrame(X_test_small)
    X['CNNqcd'] = CNNqcd
    X['CNNwz'] = CNNwz
    X['CNNtop'] = CNNtop

    # Create a label column with labels as strings ('qcd', 'wz', 'top') so that the functions used in multi_class/binary can be applied without modification
    y_test_list = np.argmax(y_test_small, axis=1)
    X['label'] = y_test_list
    X['label'].replace({0: 'qcd', 1: 'wz', 2: 'top'}, inplace=True)

    return X
