import time
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

from utils import load_yaml, make_dataset_images, save_dataset

def save_images():

    #Load configuration file:
    config_file = load_yaml('config_cnn.yaml')
    logging.info(f"Loading configuration file")

    # Load data directory and saving directory:
    saving_directory = config_file['images_directory']

    #Define number of images to save:
    nJets = config_file['nJets']
    jets = config_file['jets']
    logging.info(f'Using : {jets}')

    #Get the datasets containing the images:
    logging.info(f'Making images...')
    start = time.process_time()
    X_data, y_data = make_dataset_images(config_file, nJets, jets)
    elapsed = (time.process_time() - start)    
    logging.info(f'Images loaded It took {elapsed:.2f}s to load our {nJets} images!\nBeginning normalizing/encoding...')
    
    #Save images dataset to h5py :
    save_dataset(X_data, y_data, saving_directory, nJets)
    logging.info(f'{nJets} images have been saved to {saving_directory}')


if __name__ == '__main__':
    save_images()