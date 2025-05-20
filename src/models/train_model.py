
import sklearn
import pandas as pd 
from sklearn import ensemble
import joblib
import numpy as np
import yaml
from pathlib import Path

# Charger les hyperparamètres
params_path = Path(__file__).resolve().parents[2] / "params.yaml"
params = yaml.safe_load(open(params_path))["model"]

print(joblib.__version__)

X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

#rf_classifier = ensemble.RandomForestClassifier(n_jobs = -1)
rf_classifier = ensemble.RandomForestClassifier(
    n_estimators=params["n_estimators"],
    criterion=params["criterion"],
    n_jobs=-1
)
#--Train the model
rf_classifier.fit(X_train, y_train)

#--Save the trained model to a file
model_filename = './models/trained_model.joblib'
joblib.dump(rf_classifier, model_filename)
print("Model trained and saved successfully.")
