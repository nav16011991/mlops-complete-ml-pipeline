import os
import logging
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ensuring the logs directory exists
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
# Configuring logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, "model_training.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Remove all existing handlers associated with the logger
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.propagate = False
logger.debug(f"Process started with PID: {os.getpid()}")

def load_model(model_path: str):
    """
    Load a trained model from disk.

    Args:
        model_path (str): The path to the model file.
    Returns:
        The loaded model.
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {model_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the model on test data and return performance metrics.

    Args:
        model: The trained model to evaluate.
        X_test (np.ndarray): The test features.
        y_test (np.ndarray): The true labels for the test data.
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0)
        }
        # Remove None values from metrics
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

        logger.debug(f"Model evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: dict, output_path: str):
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics (dict): The evaluation metrics to save.
        output_path (str): The path to the output JSON file.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug(f"Metrics saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {output_path}: {e}")
        raise

def main():
    """
    Main function to load the model, evaluate it on test data, and save the metrics.
    """
    try:
        # Paths (these would typically come from a config file or environment variables)
        model_path = "models/random_forest_model.pkl"
        test_data_path = "data/processed/test_vectorized.csv"
        metrics_output_path = "reports/model_evaluation_metrics.json"

        # Load test data
        logger.debug(f"Loading test data from {test_data_path}")
        test_data = pd.read_csv(test_data_path)
        X_test = test_data.drop('label', axis=1).values
        y_test = test_data['label'].values

        # Load the trained model
        model = load_model(model_path)

        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)

        # Save the evaluation metrics
        save_metrics(metrics, metrics_output_path)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()