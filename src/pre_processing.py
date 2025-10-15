import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
import sys

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Ensuring the logs directory exists
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
# Configuring logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, "pre_processing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transform the input text by converting to lowercase, removing punctuation,
    tokenizing, removing stopwords, and applying stemming.

    Args:
        text (str): The input text to be transformed.
    Returns:

        str: The transformed text.
    """
    try:
        ps = PorterStemmer()

        # Converting text to lowercase
        text = text.lower() 

        # Tokenizing the text
        text = nltk.word_tokenize(text) 

        # Remove non-alphanumeric token
        text = [word for word in text if word.isalnum()] 

        # Remove stopwords and punctuations
        text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

        # Apply stemming
        tokens = [ps.stem(word) for word in text]
        
        # The tokens list now contains stemmed, lowercase, alphanumeric words with stopwords and punctuation removed
        transformed_text = ' '.join(tokens)
        return transformed_text
    except Exception as e:
        logger.error(f"Error transforming text: {e}")
        raise


def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocesses the DataFrame by transforming the text column and encoding the target column.
    Args:
        df (pd.DataFrame): Input DataFrame containing text and target columns.
        text_column (str): Name of the column containing text data.
        target_column (str): Name of the column containing target labels.
    Returns:
        pd.DataFrame: DataFrame with transformed text and encoded target columns.
    """
    try:
        
        logger.debug("Starting preprocessing for Dataframe")
        # Encode the target column
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
        logger.debug("Target column encoded")

        # Remove the duplicate
        df = df.drop_duplicates(keep = "first") 
        logger.debug("Duplicates removed")

        # Transform the text column
        df.loc[:, text_column] = df[text_column].astype(str).apply(transform_text)

        logger.info("Preprocessing completed successfully.")
        return df
    except KeyError as ke:
        logger.error(f"KeyError: {ke}. Please check if the specified columns exist in the DataFrame.")
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


def main(text_column= 'text', target_column= 'target'):
    """
    Main function to load raw data, preprocess it and save the preprocessed data
    """
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Data loaded properly")

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.debug(f"Processed data saved successfully at {data_path}")

    except FileNotFoundError as fnf_error:
        logger.error(f"FileNotFoundError: {fnf_error}. Please check if the CSV files exist at the specified path.")
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data {e}")
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == '__main__':
    main()