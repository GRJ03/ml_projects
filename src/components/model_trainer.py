import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest Classifier": RandomForestClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Support Vector Classifier": SVC(probability=True),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
            }

            params = {
                "Logistic Regression": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["liblinear", "lbfgs"],
                },
                "Random Forest Classifier": {
                    "n_estimators": [10, 50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                },
                "AdaBoost Classifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                },
                "Support Vector Classifier": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf"],
                },
                "K-Neighbors Classifier": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                },
                "Decision Tree Classifier": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30],
                },
            }

            # Evaluate models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            # Get the best model's name and score
            best_model_score = max(model_report.values())
            best_model_name = [
                model for model, score in model_report.items() if score == best_model_score
            ][0]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with satisfactory performance")

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Evaluate the best model on the test set
            y_test_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average="weighted")
            recall = recall_score(y_test, y_test_pred, average="weighted")
            f1 = f1_score(y_test, y_test_pred, average="weighted")

            logging.info("Model evaluation on test set:")
            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"F1 Score: {f1}")

            return {
                "best_model_name": best_model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

        except Exception as e:
            raise CustomException(e, sys)
