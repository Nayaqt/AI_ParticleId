from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

def plot_roc(y_test, predictions, result_dir, jet_types, model_type):
    """
    Plots and saves the ROC curve with AUC depending on the model type (binary or multi-class)

    Parameters : 
    - y_test (pd.df) : Test (true) labels
    - predictions (pd.series) : Predictions from the model on the test set
    - result_dir (str) : Path to results directory
    - jet_types (list) : List of the jet types, used for labelling the plots
    - model_type (str): Binary or muti_class
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    if model_type =='binary':
        # Get false and true positive rate :
        fpr, tpr, _ = roc_curve(y_test, predictions)
        # Get area under curve :
        roc_auc = auc(fpr, tpr)

        #Plot the Roc curve
        plt.plot(fpr, tpr, lw=2, label= f'QCD, AUC: {roc_auc:.2f}')
        plt.gca().set(ylabel='True positive rate', xlabel='False positive rate', title='Receiver operating characteristic for binary classification')
        plt.grid(True, which="both")
        plt.legend()
        plt.savefig(f"{result_dir}/plots/roc_curve.png")
        plt.close()
    
    #Check model type:
    if model_type == 'multi_class':
        for i, jet in enumerate(jet_types):
            # Get false and true positive rate :
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
            # Get area under curve :
            roc_auc[i] = auc(fpr[i], tpr[i])

            #Plot the curves
            plt.plot(fpr[i], tpr[i], lw=2, label= f'{jet}, AUC: {roc_auc[i]:.2f}')
            plt.gca().set(ylabel='True positive rate', xlabel='False positive rate', title='Receiver operating characteristic for multi-class data')
            plt.grid(True, which="both")
            plt.legend()
        
        #Save to directory
        plt.savefig(f"{result_dir}/plots/roc_curve.png")
        plt.close()


def plot_tagg_eff(y_test, predictions, result_dir, jet_types, model_type):
    """
    Plots and saves tagging efficiency vs background rejection. (True positive rate vs 1/false_positive_rate)
    
    Parameters : 
    - y_test (pd.df) : Test (true) labels
    - predictions (pd.series) : Predictions from the model on the test set
    - result_dir (str) : Path to results directory
    - jet_types (list) : List of the jet types, used for labelling the plots
    - model_type (str): Binary or muti_class
    
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    #Check model type:
    if model_type == 'multi_class':
        for i, jet in enumerate(jet_types):
            #Get True and False positive rates :
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
            #Get area under curve:
            roc_auc[i] = auc(fpr[i], tpr[i])
            #Plot tagging efficiency (defined as true_positive_rate vs 1/(false positive rate))
            plt.plot(tpr[i], 1/fpr[i], label=f"{jet} (Test set), AUC: {roc_auc[i]:.2f}")
            plt.gca().set(ylabel='1/False positive rate', xlabel='True positive rate', title='Tagging efficiency vs. background rejection for multi_class', xlim=(-0.01,1.01), ylim=(1,5*10**3), yscale='log')
            plt.grid(True, which="both")
            plt.legend(loc='upper right');
        #Save to results directory
        plt.savefig(f"{result_dir}/plots/eff_vs_bg_rej.png") 
        plt.close() 

    if model_type =='binary':        
        #Get True and False positive rates :
        fpr, tpr, _ = roc_curve(y_test, predictions)
        #Get area under curve:
        roc_auc = auc(fpr, tpr)
        #Plot tagging efficiency (defined as true_positive_rate vs 1/(false positive rate))
        plt.plot(tpr, 1/fpr, label=f"QCD (Test set), AUC: {roc_auc:.2f}")
        plt.gca().set(ylabel='1/False positive rate', xlabel='True positive rate', title='Tagging efficiency vs. background rejection for binary classification', xlim=(-0.01,1.01), ylim=(1,5*10**3), yscale='log')
        plt.grid(True, which="both")
        plt.legend(loc='upper right');
        #Save to results directory
        plt.savefig(f"{result_dir}/plots/eff_vs_bg_rej.png") 
        plt.close()

def to_gev(data, normalized_mass):
    """
    Computes the mass value in [GeV] before normalization using mean and std from the full dataset
    
    Parameters :
    - data (pd.df) : dataframe containing the "small" (ie: detector) dataset data.
    - normalized_mass (float) : min/max mass value, normalized

    Returns : 
    - gev_mass (float) : Mass in GeV 
    """
    mu = np.mean(data.iloc[:,3].values, axis = 0, keepdims= True)
    std = np.std(data.iloc[:,3].values, axis = 0, keepdims = True)

    gev_mass = normalized_mass*std + mu
    return gev_mass

def plot_mass_roc(model, X_test, y_test, data, jet_types, results_dir, model_type):
    """
    Computes AUC and plots the ROC curve for different mass ranges (as our efficiency can depend on the jet mass)

    Parameters :
    - model (keras model) : Model used for predictions
    - X_test (pd.df) : Test data
    - y_test (pd.series) : Labels (true) data
    - data (pd.df) : "small" (detector) data
    - jet_types (list) : list of jet_types considered
    - results_dir (str) : Path to results directory
    - model_type (str) : Type of model (binary/multi-class)
    """

    # Define binning for mass values:
    lower_bound = np.min(X_test[:,3])
    upper_bound = np.max(X_test[:,3])
    step = np.abs(upper_bound - lower_bound)/15
    bins = np.arange(lower_bound, upper_bound, step)

    #Get predictions for each mass range :
    for lower, upper in zip(bins[:-1], bins[1:]):
        X = X_test[(X_test[:,3] >= lower) & (X_test[:,3] < upper)]
        y = y_test[(X_test[:,3] >= lower) & (X_test[:,3] < upper)]
        predictions = model.predict(X)

        #Convert normalized mass values to original (GeV) values :
        lower_gev = int(to_gev(data, lower))
        upper_gev = int(to_gev(data, upper))

        #Define dicts:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        #Create and save plots depending on model_type:
        if model_type == 'binary':
            #Get True and False positive rates:
            fpr, tpr, _ = roc_curve(y, predictions)
            # Compute area under curve:
            roc_auc = auc(fpr, tpr)
            #Plot Roc Curve:
            plt.plot(fpr, tpr, lw=2, label= f' AUC: {roc_auc:.2f}')
            plt.gca().set(ylabel='True positive rate', xlabel='False positive rate', title=f'ROC curve for m in range [{lower_gev}, {upper_gev}] GeV')
            plt.grid(True, which="both")
            plt.legend()
            plt.savefig(f"{results_dir}/plots/roc_curve_m_{lower_gev}_{upper_gev}GeV.png")
            plt.close()

        if model_type =='multi_class':
            for i, jet in enumerate(jet_types):
                #Get True and False positive rates:
                fpr[i], tpr[i], _ = roc_curve(y[:, i], predictions[:, i])
                # Compute area under curve:
                roc_auc[i] = auc(fpr[i], tpr[i])
                # Plot Roc Curve:
                plt.plot(fpr[i], tpr[i], lw=2, label= f'{jet}, AUC: {roc_auc[i]:.2f}')
                plt.gca().set(ylabel='True positive rate', xlabel='False positive rate', title=f'ROC curve for m in range [{lower_gev}, {upper_gev}] GeV')
                plt.grid(True, which="both")
                plt.legend()
            plt.savefig(f"{results_dir}/plots/roc_curve_m_{lower_gev}_{upper_gev}GeV.png")
            plt.close()

        

