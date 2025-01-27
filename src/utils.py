import os
import sys

import numpy as np 
import pandas as pd
import dill

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple classification models using GridSearchCV and returns their performance scores.

    Parameters:
        X_train (DataFrame or ndarray): Training features.
        y_train (Series or ndarray): Training labels.
        X_test (DataFrame or ndarray): Testing features.
        y_test (Series or ndarray): Testing labels.
        models (dict): Dictionary of models to evaluate.
        param (dict): Dictionary of hyperparameter grids for each model.

    Returns:
        dict: A report containing test scores (accuracy, precision, recall, F1 score) for each model.
    """
    try:
        report = {}

        for model_name, model in models.items():
            params = param[model_name]

            # Perform grid search
            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)

            # Get the best parameters and update the model
            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            # Predictions
            y_test_pred = best_model.predict(X_test)

            # Compute classification metrics
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')

            # Store metrics in the report
            report[model_name] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
