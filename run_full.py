import os
import numpy as np
np.set_printoptions(precision=3)

import pandas as pd
pd.options.mode.chained_assignment = None

import time
import logging
logging.basicConfig(level=logging.INFO)

from utils import encoder, get_mass_weights, make_images_small_dataset, normalize_get_weights, load_yaml, split_data, plot_epoch_performance, read_images_hdf, normalize_reshape, cnn_encoder, split_cnn_data
from eval import plot_roc, plot_tagg_eff, plot_mass_roc
from models import classifier

def run_full():

    #Load configuration file :
    config_file = load_yaml('config_images_small.yaml')
    logging.info(f"Loading configuration file")

    # Define data/images directories :
    data_directory = config_file['small_saving_directory']
    images_directory = config_file['image_saving_directory']

    # Define results directory to store plots/model: 
    time_now = time.strftime(f"_%m-%d_%Hh%M")
    results_directory = config_file['model_name']+ '_use_CNN_pred_' + str(config_file['use_cnn_predictions']) + '_nJets_' + str(config_file['nJets']) +'_lr_' + str(config_file['NN_structure']['lr'])+ '_bs_' + str(config_file['NN_structure']['batch_size']) + '_nEpochs_' + str(config_file['NN_structure']['nEpochs']) + '_dropout_' + str(config_file['NN_structure']['dropout']) + '_batchNorm_' + str(config_file['NN_structure']['batch_normalization']) + '_' + time_now

    logging.info(f'Creating results directory under : {results_directory}')
    os.makedirs(results_directory, exist_ok = True)

    #Define model type (binary or multi_class):
    model_type = config_file['model_type']
    logging.info(f"Starting with {model_type} model.")

    # Load parameters from config_file:
    jet_types = config_file['jets']
    nJets = config_file['nJets']    

    # Load small.h5  data from file:
    logging.info(f"Preparing data using {jet_types} samples.")
    small_data = pd.read_hdf(f'{data_directory}_{nJets}.h5', 'df', 'r')
    
    #Normalize the data, split into variables and labels, computes weights to account for class imbalance/mass distribution.
    logging.info(f"Normalizing data")
    X_small, y_small, class_weights, weights = normalize_get_weights(small_data, jet_types)
    mass_weights = get_mass_weights(X_small, y_small, weights, jet_types)
    y_small, class_weights = encoder(y_small, class_weights, jet_types)   
    
    #Define training/validation/test sets:
    logging.info("Splitting the data into training/validation and testing sets.")
    X_train_small, X_test_small, X_val_small, y_train_small, y_test_small, y_val_small, W_train_small  = split_data(X_small, y_small, mass_weights)

    #If use_cnn_predictions == True, loads a new dataset with CNN predictions from a loaded model.
    if config_file['use_cnn_predictions']:

        #Load dataset containing the images:
        logging.info(f'Loading images...')
        start = time.process_time()                         
        X_data_images, y_data_images = read_images_hdf(images_directory, nJets)
        elapsed = (time.process_time() - start)
    
        logging.info(f'Images loaded It took {elapsed:.2f}s to load our {nJets} images! Reshaping the data..')

        # Normalize images:
        X_images = normalize_reshape(X_data_images)

        #Encode the target:
        y_images = cnn_encoder(y_data_images)

        logging.info(f'Splitting data into training, validation, test sets using random_state = 42')
        #Get X_test_images using the same random_state as for small_data splitting.
        _, X_test_images, _, _, _, _ = split_cnn_data(X_images,y_images)

        #Create new dataframe with CNN predictions as variables.
        data = make_images_small_dataset(X_test_images, X_test_small, y_test_small, config_file)     
 
        #Normalize the data, split into variables and labels, computes weights to account for class imbalance/mass distribution.
        logging.info(f"Normalizing data")
        X, y, class_weights, weights = normalize_get_weights(data, jet_types)
        mass_weights = get_mass_weights(X, y, weights, jet_types)
        y, class_weights = encoder(y, class_weights, jet_types)

        #Define training/validation/test sets
        logging.info("Splitting the data into training/validation and testing sets.")
        X_train, X_test, X_val, y_train, y_test, y_val, W_train  = split_data(X, y, mass_weights)      

        # Fit the model   
        model, history = classifier(X_train, y_train, W_train, X_val, y_val, config_file, results_directory, model_type)

        # Get epoch performance :
        plot_epoch_performance(history, results_directory)

        # Evaluate the model :
        logging.info(f"Getting predictions")
        predictions = model.predict(X_test)

        logging.info(f"Plotting Roc curve, tagging efficiency, etc..")    
        plot_roc(y_test, predictions, results_directory, jet_types, model_type)
        plot_tagg_eff(y_test, predictions, results_directory, jet_types, model_type)


    else:     
        # Fit the model   
        model, history = classifier(X_train_small, y_train_small, W_train_small, X_val_small, y_val_small, config_file, results_directory, model_type)
    
        # Get epoch performance :
        plot_epoch_performance(history, results_directory)

        # Evaluate the model :
        logging.info(f"Getting predictions")
        predictions = model.predict(X_test_small)

        # Create plots :
        logging.info(f"Plotting Roc curve, tagging efficiency, etc..")    
        plot_roc(y_test_small, predictions, results_directory, jet_types, model_type)
        plot_tagg_eff(y_test_small, predictions, results_directory, jet_types, model_type)

        logging.info(f'Plotting predictions vs mass..')
        plot_mass_roc(model, X_test_small, y_test_small, small_data, jet_types, results_directory, model_type)

if __name__ == '__main__':
    run_full()