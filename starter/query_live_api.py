import requests
import json

# Load environment variables from .env file

url = "https://census-classifier.onrender.com/inference"


# explicit the sample to perform inference on
sample =  {
    "age":60,
    "workclass":"Private", 
    "fnlgt":220098,
    "education":"HS-grad",
    "education-num":9,
    "marital-status":"Married-civ-spouse",
    "occupation":"Other-service",
    "relationship":"Wife",
    "race":"White",
    "sex":"Male",
    "capital-gain":0,
    "capital-loss":0,
    "hours-per-week":40,
    "native-country":"United-States"
}

data = json.dumps(sample)

# post to API and collect response
response = requests.post(url, data=data )

# display output - response will show sample details + model prediction added
print("response status code", response.status_code)
print("response content:")
print(response.json())