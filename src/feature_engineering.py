import yaml
import pandas as pd
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensuring the logs directory exists
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
# Configuring logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, "feature_engineering.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(config_path: str) -> dict:
    """
    Load parameters from a YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.
    Returns:
        dict: The parameters loaded from the YAML file.
    """
    try:
        with open(config_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.debug(f"Parameters loaded successfully from {config_path}")
            return params
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading parameters: {e}")
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        data_url (str): The URL or path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(data_url)
        df.fillna('', inplace=True)
        logger.debug(f"Data loaded and NaNs filled successfully from {data_url}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while loading data from {data_url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {data_url}: {e}")
        raise

def apply_tfidf_vectorization(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Apply TF-IDF vectorization to the text data.

    Args:
        train_data (pd.DataFrame): The training data containing a 'text' column.
        test_data (pd.DataFrame): The testing data containing a 'text' column.
        max_features (int): The maximum number of features for TF-IDF.

    Returns:
        tuple: A tuple containing the transformed training and testing feature matrices.
    """
    try:
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        X_test = test_data['text'].values
        y_train = train_data['target'].values
        y_test = test_data['target'].values

        X_train_bow = tfidf_vectorizer.fit_transform(X_train)
        X_test_bow = tfidf_vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test
        
       
        logger.debug("TF-IDF vectorization applied to testing data.")

        return train_df, test_df
    except Exception as e:
        logger.error(f"Error during TF-IDF vectorization: {e}")
        raise
    

def save_data( df: pd.DataFrame, data_path: str) -> None:   
    """
    Save the data to CSV files.

        Args:
            df (pd.DataFrame): The  data frame.
            data_path (str): The path to save the data CSV.
            
    """
    try:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.debug(f"Vectorized data saved successfully at {data_path}")
    except Exception as e:
        logger.error(f"Error saving vectorized data: {e}")
        raise

def main():
    try:
        # Fetch the data from data/interim
        train_data = load_data("./data/interim/train_processed.csv")
        test_data = load_data("./data/interim/test_processed.csv")
        logger.debug("Data loaded properly")

        # Apply TF-IDF vectorization
        # max_features = 50  # You can adjust this value as needed

        params = load_params("params.yaml")
        max_features = params['feature_engineering']['max_features']

        train_vectorized, test_vectorized = apply_tfidf_vectorization(train_data, test_data, max_features)

        # Store the vectorized data inside data/processed
        save_data(train_vectorized, os.path.join("./data", "processed","train_vectorized.csv"))
        save_data(test_vectorized, os.path.join("./data", "processed","test_vectorized.csv"))

    except FileNotFoundError as fnf_error:
        logger.error(f"FileNotFoundError: {fnf_error}. Please check if the CSV files exist at the specified path.")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == '__main__':
    main()