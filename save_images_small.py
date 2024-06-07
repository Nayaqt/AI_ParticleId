import time
import logging
import pandas as pd
pd.options.mode.chained_assignment = None
logging.basicConfig(level=logging.INFO)

from utils import load_yaml, make_dataset_images, prepare_dataset, save_dataset

def save_images_small():

    #Load configuration file:
    config_file = load_yaml('config_images_small.yaml')
    logging.info(f"Loading configuration file")

    # Data directory:
    data_directory = config_file['data_directory']

    # Load data_small saving directory and image saving directory:
    image_saving_directory = config_file['image_saving_directory']
    small_saving_directory = config_file['small_saving_directory']

    #Define number of images and flavors to save:
    jet_types = config_file['jets']
    nJets = config_file['nJets']
    logging.info(f'Total number of jets: {nJets*len(jet_types)} \nJets flavours used {jet_types}')

    #Get the datasets containing the images:
    logging.info(f'Making images...')
    start = time.process_time()
    X_data_images, y_data_images = make_dataset_images(config_file, nJets, jet_types)
    elapsed = (time.process_time() - start)    
    logging.info(f'Images loaded It took {elapsed:.2f}s to load our {nJets} images!\nBeginning normalizing/encoding...')
    
    #Get dataset from small.h5:
    data_small = prepare_dataset(data_directory, jet_types, nJets)
    
    #Save images dataset to h5 file :
    logging.info(f'Saving images..')
    save_dataset(X_data_images, y_data_images, image_saving_directory, nJets)
    logging.info(f'{nJets} images have been saved to {image_saving_directory}')    

    #Save data_small to h5 file:
    logging.info(f'Saving small_data to {small_saving_directory}{nJets}.h5')
    data_small.to_hdf(f'{small_saving_directory}_{nJets}.h5', key = 'df', mode = 'w')

if __name__ == '__main__':
    save_images_small()