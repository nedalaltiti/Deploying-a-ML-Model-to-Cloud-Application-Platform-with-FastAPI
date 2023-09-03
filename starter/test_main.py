from fastapi.testclient import TestClient
import json
import logging
from main import app


client = TestClient(app)


def test_say_welcome():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting" : "Welcome!"}


def test_high_predictions():
    response = client.post("/inference/", 
                json={"age": 37,
                      "workclass": "Private",
                      "fnlgt": 280464,
                      "education": "Some-college",
                      "education-num": 10,
                      "marital-status": "Married-civ-spouse",
                      "occupation": "Exec-managerial",
                      "relationship": "Husband",
                      "race": "Black",
                      "sex": "Male",
                      "capital-gain": 0,
                      "capital-loss": 0,
                      "hours-per-week": 80,
                      "native-country": "United-States"})
    
    assert response.status_code == 200, "Expected status code 200"
    assert response.json()["salary"] == '<=50K', "Expected salary to be higher than 50K"

def test_low_predictions():
    response = client.post("/inference/",
                json={"age": 50,
                      "workclass": "Private",
                      "fnlgt": 176609,
                      "education": "Some-college",
                      "education-num": 10,
                      "marital-status": "Divorced",
                      "occupation": "Other-service",
                      "relationship": "Not-in-family",
                      "race": "White",
                      "sex": "Male",
                      "capital-gain": 0,
                      "capital-loss": 0,
                      "hours-per-week": 45,
                      "native-country": "United-States"})

    assert response.status_code == 200, "Expected status code 200"
    assert response.json()["salary"] == "<=50K", "Expected salary to be less than or equal 50K"


def test_wrong_inference_query():
    """
    Test incomplete sample does not generate prediction
    """
    sample =  {  'age':50,
                'workclass':"Private", 
                'fnlgt':234721,
            }

    data = json.dumps(sample)
    response = client.post("/inference/", data=data )

    assert 'prediction' not in response.json().keys()
    logging.warning(f"The sample has {len(sample)} features. Must be 14 features")    