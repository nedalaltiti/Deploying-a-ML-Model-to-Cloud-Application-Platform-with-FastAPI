# Model Card
Model Developed by: Nedal Altiti
Date: 29 / 08 / 2023
Version: 1.0.0
Type: Binary Clasifier
Dataset Used: https://archive.ics.uci.edu/ml/datasets/census+income

## Model Details
The model is a Random Forest Classifier with 200 estimators and a maximum depth of 5. Its hyperparameters have been tuned using the Grid Search - CV technique.


## Intended Use
The model's objective is to predict if the salary/income of an individual is above or below a threshold of 50k $, given fifteen different variables:

age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


## Training Data
The model was trained on a labeled dataset containing information about individuals from the Census dataset. The features include age, education level, occupation, work class, etc., and the target variable is binary, indicating whether the individual earns more than $50,000 per year.

The dataset was preprocessed and split into training and validation sets to evaluate the model's performance. 
## Evaluation Data
The datasets used in the project were the Census Income Data Set from UCI Machine Learning Repository. A binary classification model was designed to determine whether or not a person's income is over 50K based on several features. Preprocessing: (can be checked in EDA.ipynb)

20% of total samples
Drop NaN
Remove all extra space in string - All categorical column are encoded using one hot encoding
All categorical column are encoded using OneHotEncoder from scikit-learn
Label column ('salary') is encoded using LabelBinarizer from scikit-learn
## Metrics
The metrics the model have been evaluated on are the F-score, precision and recall. Therefore, the trained and fine-tuned model obtained the following score on the respective metrics:

Precision: 0.823
Recall: 0.570
Fbeta: 0.674
Confusion matrix:
                    [[4753  192]
                     [ 674  894]]
## Ethical Considerations
There is no sensitive information. There is no use of the data to inform decisions about matters important to human well-being - such as health or safety.
## Caveats and Recommendations
The data is biased based on gender. Have data imbalance that need to be investigated.