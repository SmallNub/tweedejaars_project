from pathlib import Path
import pandas as pd
import pickle

from ..config import PROCESSED_DATA_DIR, MAIN_DATA_FILE_NAME, MODELS_DIR


def load_df(file_path: Path = PROCESSED_DATA_DIR / MAIN_DATA_FILE_NAME):
    """Load the dataframe from a pickle file."""
    df = pd.read_pickle(file_path)
    return df


def save_df(
    df: pd.DataFrame,
    file_path: Path = PROCESSED_DATA_DIR / MAIN_DATA_FILE_NAME
):
    """Save the dataframe to a pickle file."""
    pd.to_pickle(df, file_path)


def load_model(file_name: str):
    """Load a model from a pickle file."""
    file_path = MODELS_DIR / file_name
    return pickle.load(file_path)


def save_model(file_name: str):
    """Save a model to a pickle file."""
    file_path = MODELS_DIR / file_name
    pickle.dump(file_path)
