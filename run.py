import os

import numpy as np
np.set_printoptions(precision=3)

import pandas as pd
pd.options.mode.chained_assignment = None

import time
import logging
logging.basicConfig(level=logging.INFO)

from utils import encoder, get_mass_weights, normalize_get_weights, load_yaml, prepare_dataset, split_data, plot_epoch_performance
from eval import plot_roc, plot_tagg_eff, plot_mass_roc
from models import classifier

def run():
    #Load configuration file :
    config_file = load_yaml('config.yaml')
    logging.info(f"Loading configuration file")

    # Load the data into memory
    jet_types = config_file['jets']
    nJets = config_file['nJets']

    # Data directory :
    data_dir = config_file['data_directory']

    #Results directory : 
    time_now = time.strftime(f"_%m-%d_%Hh%M")
    results_directory = config_file['model_name'] + '_nJets_' + str(jet_types[1]) + '_' + str(config_file['nJets']) +'_lr_' + str(config_file['NN_structure']['lr'])+ '_bs_' + str(config_file['NN_structure']['batch_size']) + '_epochs_' + str(config_file['NN_structure']['nEpochs']) + '_dropout_' + str(config_file['NN_structure']['dropout']) + '_batchNorm_' + str(config_file['NN_structure']['batch_normalization']) + '_' + time_now

    logging.info(f'Creating results directory under : {results_directory}')
    os.makedirs(results_directory, exist_ok = True)

    #Model type (binary/multiclass):
    model_type = config_file['model_type']
    logging.info(f"Starting with {model_type} model.")

    #Prepares the dataset, merges the different files and adds the corresponding label.
    logging.info(f"Preprocessing data using {jet_types} samples")
    data = prepare_dataset(data_dir, jet_types, nJets)    
    
    #Normalize the data, split into variables and labels, computes weights to account for class imbalance/mass distribution.
    logging.info(f"Normalizing data")
    X, y, class_weights, weights = normalize_get_weights(data, jet_types = jet_types)
    mass_weights = get_mass_weights(X, y, weights, jet_types)
    y, class_weights = encoder(y, class_weights, jet_types)   
    
    #Define training/validation/test sets
    logging.info("Splitting the data into training/validation and testing sets.")
    X_train, X_test, X_val, y_train, y_test, y_val, W_train  = split_data(X, y, mass_weights)      

    # Fit the model
    logging.info(f"Training started, using {len(X_train)} jets for training.")
    model, history = classifier(X_train, y_train, W_train, X_val, y_val, config_file, results_directory, model_type)
    
    # Get epoch performance :
    plot_epoch_performance(history, results_directory)

    # Evaluate the model :
    logging.info(f"Getting predictions")
    predictions = model.predict(X_test)

    # Create plots:
    logging.info(f"Plotting Roc curve, tagging efficiency, etc..")    
    plot_roc(y_test, predictions, results_directory, jet_types, model_type)
    plot_tagg_eff(y_test, predictions, results_directory, jet_types, model_type)

    logging.info(f'Plotting predictions vs mass..')
    plot_mass_roc(model, X_test, y_test, data, jet_types, results_directory, model_type)

if __name__ == '__main__':
    run()