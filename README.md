# How to run the code
<details>
<summary>Show/Hide Table of Contents</summary>

[[_TOC_]]

</details>

## Overview

This text file describes how to run the code depending if you want to train a binary or multi_class classifier using rather a classical fully connected network or a CNN using jet images. You can also choose to run a multi_class network with CNN predictions as additional input to small.h5 variables.

This code uses configuration files which you have to make sure you setup according to your needs before running the code.

## Fully connected network
### Configuration file
If you choose to run a fully connected network, the corresponding configuration file is "config.yaml".
This file contains several options which you can modify :

* model_name : Name of the model to use for saving, good options are 'multi_class', 'binary'.

* data_directory : Path to data directory, default is 'data'.

* model_type : Either 'multi_class' or 'binary'. This will determine if you are training on a multi_class or binary classifier. Basically modifies the shape of the output, output activation and loss function as well as the architecture which is somehow optimized for the task.

* jets : list containing jet flavors as strings, e.g: ['qcd','wz','top']. Be careful to respect the order defined for correct labelling.

* nJets : Number of jets per class to use, fill nothing to use all available jets.

* NN_structure : This is where you can set the parameters for the network, from learning rate to architecture.
    - lr : 0.0001
    - batch_size : 1000
    - nEpochs : 50
    - dropout : 0.2
    - batch_normalization : False
    - nClasses : 3

* Structure of the dense layers : 
    - input_layer : 64
    - dense_layers : [64, 32, 16]

If further modifications are needed, `models.py` contains the definition of the model under the classifier function. Here you can decide whether or not to use callbacks such as ReduceLROnPlateau, ModelCheckpoint and EarlyStopping, each with their own parameters.

### Actually running the code
Once the configuration file is set, you can run the code from the terminal by using :
`python run.py`
Logging will display the status of the code in the terminal.

This will run the code, i.e : 
* Load the data into a dataframe
* Normalize the features and encode the target depending on the number of classes
* Split the data into training/validation/test sets
* Fit the model
* Get predictions from the model on the test set
* Produce plots in a folder created in the form of "multi_class_lr_0.0005_bs_128_epoch_50_dropout_0.2_batch_norm_no_05-17_09h30/plots"
* The created plots are the following :
    - loss evolution : monitor the training and validation loss at each epoch during training
    - accuracy evolution : (same)
    - ROC_curve : False positive rate vs true positive rate: one of the main curve to check the performance of the model
    - ROC_curve mass dependancy : Shows different plots depending on the mass range
    - eff vs bg curve : Usefull in jet tagging: shows the tagging efficiency vs background rejection

## CNN
### Saving images to directory
Because it can take a long time to load images from const.h5 file, it is best to save them to a directory using the `save_images.py` script.
In order to do so, make sure to set up the following parameters in the configuration file 'config_cnn.yaml':
* images_directory
* data_directory
* jets
* nJets

Description of how to use these parameters is given in the 'Configuration file' section below.
Once this is done, images can be saved to file using:
`python save_images.py`

### Configuration file
If you choose to run a Convolutional neural network (CNN), the corresponding configuration file is `config_cnn.yaml`.
This file contains several options which you can modify :

* model_name : Name of the model to use for saving, one good option is 'CNN'.

* model_type : Type of classifier to train, either 'binary' or 'multi_class'.

* data_directory : Path to directory containing const.h5 data. Default is 'data'.

* images_directory : Path to directory containing image dataset saved using `save_images.py` script.

* jets : list containing jet flavors as strings, e.g: ['qcd','wz','top']. Be careful to respect the order defined for correct labelling.

*nJets : Number of jets per class to use.

* NN_structure : This is where you can set the parameters for the network, from learning rate to architecture.
    - lr : 0.0001
    - batch_size : 1000
    - nEpochs : 50
    - dropout : 0.2
    - batch_normalization : False
    - nClasses : 3

* Structure of the Convolutional layers : 
    - filter_size : Size of the filters in the first 2 convolutional layers.
    - kernel_size : Size of the kernel in the first 2 convolutional layers, must be an odd number.

If further modifications are needed, `models.py` contains the definition of the model under the make_cnn function. Here you can decide whether or not to use callbacks such as ReduceLROnPlateau, ModelCheckpoint and EarlyStopping, each with their own parameters.

### Running the code:
Once the configuration file is set, and the images saved using `save_images.py` you can run the code from the terminal by using :
 `python run_cnn.py` 
Logging will display the status of the code in the terminal.

This will run the code, i.e : 
* Load the images from file
* Normalize the 'pixels' and encode the target depending on the number of classes
* Split the data into training/validation/test sets
* Fit the model
* Get predictions from the model on the test set
* Produce plots in a folder created in the form of "cnn_lr_0.0005_bs_128_epoch_50_dropout_0.2_batch_norm_no_05-17_09h30/plots"
* The created plots are the following :
    - loss evolution : monitor the training and validation loss at each epoch during training
    - accuracy evolution : (same)
    - ROC_curve : False positive rate vs true positive rate: one of the main curve to check the performance of the model
    - eff vs bg curve : Usefull in jet tagging: shows the tagging efficiency vs background rejection

## Using CNN predictions as inputs.
In order to use CNN predictions as inputs to a feedforward NN using the small.h5 variables, the procedure is similar to the one described for running the CNN.

### Saving images to directory
Because it can take a long time to load images from const.h5 file, it is best to save them to a directory using the `save_images_small.py`script. This script differs from the `save_images.py` script in the fact that it also saves the data from small.h5. This was done to ensure that both datafiles are saved in the same way.
In order to do so, make sure to set up the following parameters in the configuration file "config_images_small.yaml":
* image_saving_directory 
* small_saving_directory 
* jets 
* nJets

Description of how to define these parameters is given in the 'Configuration file' section below.
Once this is done, images can be saved to file using:
`python save_images_small.py`

### Configuration file
If you choose to use CNN predictions as inputs, the corresponding configuration file is "config_images_small.yaml".
This file contains several options which you can modify :

* model_name : Name of the model to use for saving, one good option is 'multi_class_cnn'.

* model_type : Type of classifier to train, either 'binary' or 'multi_class'. Only 'multi_class' is supported with CNN predictions.

* data_directory : Path to directory containing const.h5 data. Default is 'data'.

* images_saving_directory : Path to directory containing image dataset saved using `save_images_small.py` script.

* small_saving_directory : Path to directory containing small dataset saved using `save_images_small.py` script.

* use_cnn_predictions : Boolean which determines if the predictions from CNN are to be used.

* cnn_model : Path to saved CNN_model, if none is saved you must add `model_chkpt` as callbacks in `models.py` when running the CNN.

* jets : list containing jet flavors as strings, e.g: ['qcd','wz','top']. Be careful to respect the order defined for correct labelling.

*nJets : Number of jets per class to use. Note: if nothing is filled, it will (try to) save all the available jets (around 1.5e6). Because of my computer, the max number of jet per class that could be loaded is 100_000. The code should work with more but has not been tested.

* NN_structure : This is where you can set the parameters for the network, from learning rate to architecture.
    - lr : 0.0001
    - batch_size : 1000
    - nEpochs : 50
    - dropout : 0.2
    - batch_normalization : False
    - nClasses : 3

* Structure of the dense layers : 
    - input_layer : 256
    - dense_layers : [128, 64, 32, 30]

### Running the code

Once the configuration file is set, and the images saved using `save_images_small.py` you can run the code from the terminal by using :
 `python run_full.py` 
Logging will display the status of the code in the terminal.

This will run the code, i.e : 
* Load the images and dataset from file
* Normalize the 'pixels' and encode the target depending on the number of classes
* Split the data into training/validation/test sets
* Fit the model
* Get predictions from the model on the test set
* Produce plots in a folder created in the form of "multi_class_cnn_use_CNN_pred_False_nJets_100000_lr_0.0001_bs_32_nEpochs_100_dropout_0.2_batchNorm_False__05-31_07h26/plots"
* The created plots are the following :
    - loss evolution : monitor the training and validation loss at each epoch during training
    - accuracy evolution : (same)
    - ROC_curve : False positive rate vs true positive rate: one of the main curve to check the performance of the model
    - Roc_curve mass dependency : Shows the AUC scores for different mass ranges.
    - eff vs bg curve : Usefull in jet tagging: shows the tagging efficiency vs background rejection


 


