"""
This module contains the training, metrics and inference of machine learning model.

Author: Nedal Altiti
Date: 23 / 08 / 2023
"""
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import logging
import multiprocessing


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    parameters = {
        'n_estimators': [10, 50, 100],
        'max_depth': [5, 10],
    }

    njobs = multiprocessing.cpu_count() - 1
    logging.info("Searching best hyperparameters on {} cores".format(njobs))


    rf = GridSearchCV(RandomForestClassifier(random_state=42),
                       param_grid=parameters,
                       cv=3,
                       n_jobs=njobs,
                       verbose=2,
                       )
    
    rf = RandomForestClassifier(random_state = 42)
    rf.fit(X_train, y_train)
    logging.info("********* Best parameters found ***********")
    logging.info("BEST PARAMS: {}".format(rf.best_params_))

    return rf



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    if not model:
        raise ValueError("Model RandomForestClassifier is none")
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("Model is not type of RandomForestClassifier")
    y_pred = model.predict(X)
    return y_pred

def compute_confusion_matrix(y, y_pred, labels=None):
    """
    Compute confusion matrix using the predictions and ground thruth provided

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    ------
    cm : confusion matrix for the provided prediction set
    """
    cm = confusion_matrix(y, y_pred)
    return cm

def compute_score_per_slice(test_df, categorical_feature, y, y_pred):
    """
    Compute the performance on slices for a given categorical feature
    a slice corresponds to one value option of the categorical feature analyzed
    ------
    test_df: 
        test dataframe pre-processed with features as column used for slices
    categorical_feature:
        feature on which to perform the slices
    y : np.array
        corresponding known labels, binarized.
    y_pred : np.array
        Predicted labels, binarized

    Returns
    ------
    Dataframe with
        n_samples: integer - number of data samples in the slice
        precision : float
        recall : float
        fbeta : float
    row corresponding to each of the unique values taken by the feature (slice)
    """    
    slice_options = test_df[categorical_feature].unique().tolist()
    perf_df = pd.DataFrame(index=slice_options, 
                            columns=['categorical_feature','n_samples','precision', 'recall', 'fbeta'])
    for option in slice_options:
        slice_mask = test_df[categorical_feature]==option

        slice_y = y[slice_mask]
        slice_preds = y_pred[slice_mask]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)
        
        perf_df.at[option, 'categorical_feature'] = categorical_feature
        perf_df.at[option, 'n_samples'] = len(slice_y)
        perf_df.at[option, 'precision'] = precision
        perf_df.at[option, 'recall'] = recall
        perf_df.at[option, 'fbeta'] = fbeta

    # reorder columns in performance dataframe
    perf_df.reset_index(names='feature variable', inplace=True)
    colList = list(perf_df.columns)
    colList[0], colList[1] =  colList[1], colList[0]
    perf_df = perf_df[colList]

    return perf_df    
    


