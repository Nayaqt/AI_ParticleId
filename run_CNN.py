import time
import os
import logging
logging.basicConfig(level=logging.INFO)


from utils import load_yaml, read_images_hdf, plot_epoch_performance, cnn_encoder, normalize_reshape, split_cnn_data
from models import make_cnn
from eval import  plot_roc, plot_tagg_eff

def run_CNN():

    #Load configuration file :
    config_file = load_yaml('config_cnn.yaml')
    logging.info(f"Loading configuration file")
    
    # Model type:
    model_type = config_file['model_type']

    # Number of jets to use and jet types (qcd,wz,top):
    nJets = config_file['nJets']
    jet_types = config_file['jets']
    logging.info(f'Number of jets per class: {nJets}, jets flavors used {jet_types}')
    
    #Data directory:
    data_directory = config_file['images_directory']

    #Create result directory : 
    time_now = time.strftime(f"_%m-%d_%Hh%M")
    results_directory = config_file['model_name'] + '_' + str(config_file['model_type']) + '_nJets_' + str(config_file['nJets'])+ '_' + jet_types[1] + '_' + '_lr_' + str(config_file['NN_structure']['lr'])+ '_bs_' + str(config_file['NN_structure']['batch_size']) + '_epochs_' + str(config_file['NN_structure']['nEpochs']) + time_now

    logging.info(f'Creating results directory under : {results_directory}')
    os.makedirs(results_directory, exist_ok = True)

    #Load dataset containing the images:
    logging.info(f'Loading images...')
    start = time.process_time() # Monitor loading time
    X_data, y_data = read_images_hdf(data_directory, nJets)
    elapsed = (time.process_time() - start)
    
    logging.info(f'Images loaded It took {elapsed:.2f}s to load our {nJets} images! Reshaping the data..')

    # Normalize images:
    X = normalize_reshape(X_data)

    #Encode the target:
    y = cnn_encoder(y_data)

    logging.info(f'Splitting data into training, validation, test sets using random_state = 42')
    #Split the data :
    X_train, X_test, X_val, y_train, y_test, y_val = split_cnn_data(X,y)

    # Create, compile and fit the model :
    logging.info(f'Start training. {len(X_train)} jets used for training, {len(X_val)} for validation.')
    model, history = make_cnn(X_train, y_train, X_val, y_val, config_file, model_type, results_directory)

    # Get epoch performance :
    logging.info(f'Plotting epoch performance..')
    plot_epoch_performance(history, results_directory)

    # Evaluate the model :
    logging.info(f"Getting predictions")
    predictions = model.predict(X_test)

    logging.info(f"Plotting predictions, Roc curve, etc..")    
    #visualise_predictions(predictions, y_test, results_directory)
    plot_roc(y_test, predictions, results_directory, jet_types, model_type)
    plot_tagg_eff(y_test, predictions, results_directory, jet_types, model_type)
    

if __name__ == '__main__':
    run_CNN()