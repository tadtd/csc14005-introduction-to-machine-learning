import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing the dataset.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded dataset.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


