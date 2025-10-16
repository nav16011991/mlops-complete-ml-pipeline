import os
import logging
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Ensuring the logs directory exists
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
# Configuring logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, "model_building.log")
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

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while loading data from {file_path}: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier and save the model to disk.

    Args:
        X_train (np.ndarray): The training feature matrix.
        y_train (np.ndarray): The training target vector.
        params (dict): A dictionary of hyperparameters 

    Returns:
        RandomForestClassifier: The trained RandomForestClassifier model.
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be equal.")
        
        logger.debug(f"Initializing RandomForestClassifier with parameters: {params}")

        clf = RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])

        logger.debug(f"Starting model training with {X_train.shape[0]} samples")
        clf.fit(X_train, y_train)
        logger.debug("Model training completed successfully")

        return clf
    except ValueError as e:
        logger.error(f"Value error during model training: {e}")
        raise
    except Exception as e:
        logger.error(f"Error training or saving the model: {e}")
        raise

def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """
    Save the trained model to disk.

    Args:
        model (RandomForestClassifier): The trained model.
        model_path (str): The path to save the model.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved successfully at {model_path}")
    except FileNotFoundError as fnf_error:
        logger.error(f"FileNotFoundError: {fnf_error}. Please check if the directory exists.")
        raise
    except Exception as e:
        logger.error(f"Error saving the model: {e}")
        raise

def main():
    """
    Main function to load data, train the model, and save the trained model.
    """
    try:
        # Define model parameters
        params = {
            'n_estimators': 100,
            'random_state': 42
        }
        
        # Load the training data
        train_data_path = "./data/processed/train_vectorized.csv"
        train_data = load_data(train_data_path)

        # Separate features and target
        X_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values

        

        # Train the model
        model = train_model(X_train, y_train, params)

        # Save the trained model
        model_path = "./models/random_forest_model.pkl"
        save_model(model, model_path)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()