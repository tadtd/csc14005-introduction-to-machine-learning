import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from an Excel file.

    Parameters:
    file_path (str): The path to the Excel file containing the dataset.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded dataset.
    """
    try:
        data = pd.read_excel(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise RuntimeError(f"Failed to load data from {file_path}") from e


