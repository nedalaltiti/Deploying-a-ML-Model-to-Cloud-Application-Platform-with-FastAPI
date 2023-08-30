import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from .model import compute_model_metrics, inference
from .data import process_data
import os, logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the file path to the data file
data_file_path = os.path.join(current_dir, "../../data/census_no_spaces.csv")

@pytest.fixture(scope='module')
def data():
    return pd.read_csv(data_file_path)


@pytest.fixture(scope="module")
def cat_features():
    cat_features = ["workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country"]
    return cat_features


@pytest.fixture(scope='module')
def training_data(data, cat_features):
    train, test = train_test_split( data, 
                                test_size=0.20, 
                                random_state=42, 
                                stratify=data['salary']
                                )
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True)
    return X_train, y_train


def test_import_data(data):
    """
    Test presence and shape of dataset file
    """
    # Check the df shape
    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0

    except AssertionError as err:
        logging.error(
        "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_compute_model_metrics(training_data):
    X, y = training_data
    model = pickle.load(open('../../model/trained_model.pkl', 'rb'))
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

def test_inference(training_data):
    X, y = training_data
    model = pickle.load(open('../../model/trained_model.pkl', 'rb'))
    y_pred = inference(model, X)
    assert y.shape == y_pred.shape and np.unique(y_pred).tolist() == [0, 1]

       
