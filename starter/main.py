"""
This script for Rest API
"""
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field
import pandas as pd
from starter.ml.data import process_data
import logging, pickle
import os
import json
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

current_dir = os.path.dirname(os.path.abspath(__file__))

# Instantiate the app.
app = FastAPI(title="Inference API",
              description="API takes a sample and runs as inference",
              version="1.0.0")

# Define a GET on the specified endpoint.
@app.get("/")
async def say_welcome():
    return {"greeting" : "Welcome!"}


# Declare the data object with its components and their type.
class InputData(BaseModel):
    age: int = Field(example=40)
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example=184018)
    education: str = Field(example="Assoc-voc")
    education_num: int = Field(alias="education-num", example=11)
    marital_status: str = Field(alias="marital-status", example="Married-civ-spouse")
    occupation: str = Field(example="Machine-op-inspct")
    relationship: str = Field(example="Husband")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(alias="capital-gain", example=0)
    capital_loss: int = Field(alias="capital-loss", example=0)
    hours_per_week: int = Field(alias="hours-per-week", example=38)
    native_country: str = Field(alias="native-country", example="United-States")



@app.post("/inference/")
async def predict(inference: InputData):
    X_test = pd.DataFrame(inference.dict(), index=[0])

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    model_path = os.path.join(current_dir, "model/trained_model.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
      

    encoder_path = os.path.join(current_dir, "model/encoder.pkl")
    with open(encoder_path, "rb") as file:
        encoder = pickle.load(file)

    labelizer_path = os.path.join(current_dir, "model/labelizer.pkl")
    with open(labelizer_path, "rb") as file:
        lb = pickle.load(file)

    X_test, _, _, _ = process_data(
        X_test, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
    )
    
    preds = model.predict(X_test)    
    # Convert prediction to label and add to data output
    if preds[0] > 0.5:
        salary = '>50K'
    else:
        salary = '<=50K'
    
    inference.dict()['salary'] = salary
    
    # Convert numpy.int64 to int
    inference_dict = {k: int(v) if isinstance(v, np.int64) else v for k, v in inference.dict().items()}
    
    inference_dict['salary'] = salary
    return inference_dict
    # return Response(content=json.dumps(preds.tolist()), media_type="application/json")


