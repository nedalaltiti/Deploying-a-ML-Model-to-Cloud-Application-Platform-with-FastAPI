"""
Script to train machine learning model.

Authoir: Nedal Altiti
Date: 23 / 08/ 2023
"""

# Add the necessary imports for the starter code.
import pandas as pd
import numpy as np
import os, pickle, logging
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, compute_confusion_matrix, compute_score_per_slice

def remove_if_exists(filename):
    """
    Delete a file if it exists.
    input:
        filename: str - path to the file to be removed
    output:
        None
    """
    if os.path.exists(filename):
        os.remove(filename)


# Initialize logging
logging.basicConfig(filename='journal.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Add code to load in the data.
def train_and_save_model(X_train, y_train, savepath):
    """
    Train a machine learning model and save it to disk.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        savepath (str): Path to save the model.
    """
    model = train_model(X_train, y_train)
    # Save model to disk
    with open(os.path.join(savepath, 'trained_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to disk: {savepath}")

def load_model(savepath):
    """
    Load a trained model from disk.
    
    Args:
        savepath (str): Path to the saved model.
        
    Returns:
        model: Loaded machine learning model.
    """
    with open(os.path.join(savepath, 'trained_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    return model    

datapath = "../data/census_no_spaces.csv"
data = pd.read_csv(datapath)

# Optional enhancement, use K-fold cross validation instead of a
# train-test split using stratify due to class imbalance
train, test = train_test_split( data, 
                                test_size=0.20, 
                                random_state=42, 
                                stratify=data['salary']
                                )

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Proces the test data with the process_data function.
# Set train flag = False - We use the encoding from the train set
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)
# check if trained model already exists
savepath = '../model'
filename = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

# if saved model exits, load the model from disk
if os.path.isfile(os.path.join(savepath,filename[0])):
        model = pickle.load(open(os.path.join(savepath,filename[0]), 'rb'))
        encoder = pickle.load(open(os.path.join(savepath,filename[1]), 'rb'))
        lb = pickle.load(open(os.path.join(savepath,filename[2]), 'rb'))

# Else Train and save a model.
else:
    model = train_model(X_train, y_train)
    # save model  to disk in ./model folder
    pickle.dump(model, open(os.path.join(savepath,filename[0]), 'wb'))
    pickle.dump(encoder, open(os.path.join(savepath,filename[1]), 'wb'))
    pickle.dump(lb, open(os.path.join(savepath,filename[2]), 'wb'))
    logging.info(f"Model saved to disk: {savepath}")


# evaluate trained model on test set
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

logging.info(f"Classification target labels: {list(lb.classes_)}")
logging.info(
    f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")

cm = compute_confusion_matrix(y_test, preds, labels=list(lb.classes_))

logging.info(f"Confusion matrix:\n{cm}")

# Compute performance on slices for categorical features
# save results in a new txt file
slice_savepath = "./slice_output.txt"
remove_if_exists(slice_savepath)

# iterate through the categorical features and save results to log and txt file
for feature in cat_features:
    performance_df = compute_score_per_slice(test, feature, y_test, preds)
    performance_df.to_csv(slice_savepath,  mode='a', index=False)
    logging.info(f"Performance on slice {feature}")
    logging.info(performance_df)



unique_values, counts = np.unique(y_test, return_counts=True)