import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split

# Ensuring the logs directory exists
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

# Configuring logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, "data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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
        logger.debug(f"Data loaded successfully from {data_url}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while loading data from {data_url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {data_url}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    try:
        initial_shape = df.shape
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True) # Dropping unnecessary columns
        df.rename(columns = {'v1':'target', 'v2':'text'}, inplace=True) # Renaming columns for better understanding
        final_shape = df.shape
        logger.debug(f"Data preprocessed successfully. Initial shape: {initial_shape}, Final shape: {final_shape}")
        return df
    except KeyError as e:
        logger.error(f"Missing column in dataframe while preprocessing: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise

def save_data( train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save the training and testing data to CSV files.

    Args:
        train_data (pd.DataFrame): The training data.
        test_data (pd.DataFrame): The testing data.
        data_path (str): The path to save the data CSV.
        
    """
    try:
        raw_data_dir = os.path.join(data_path, "raw")
        os.makedirs(raw_data_dir, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_dir, "train.csv"), index=False)
        logger.debug(f"Training data saved successfully at {raw_data_dir}")
        test_data.to_csv(os.path.join(raw_data_dir, "test.csv"), index=False)
        logger.debug(f"Testing data saved successfully at {raw_data_dir}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


def main():
    try:
        test_size = 0.2
        data_url = "https://raw.githubusercontent.com/nav16011991/mlops-complete-ml-pipeline/refs/heads/main/experiments/spam.csv"
        data_path = "./data"
        # Load data
        df = load_data(data_url)
        
        # Preprocess data
        final_df = preprocess_data(df)
        
        # Split data into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        logger.debug(f"Data split into training and testing sets. Training shape: {train_data.shape}, Testing shape: {test_data.shape}")
        
        # Save the processed data
        save_data(train_data, test_data, data_path)
        logger.info("Data ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()