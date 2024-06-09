from pathlib import Path
import pandas as pd
import pickle

from ..config import PROCESSED_DATA_DIR, MAIN_DATA_FILE_NAME, MODELS_DIR


def load_df(file_path: Path = PROCESSED_DATA_DIR / MAIN_DATA_FILE_NAME) -> pd.DataFrame:
    """Load the dataframe from a pickle file."""
    df = pd.read_pickle(file_path)
    return df


def save_df(df: pd.DataFrame, file_path: Path = PROCESSED_DATA_DIR / MAIN_DATA_FILE_NAME):
    """Save the dataframe to a pickle file."""
    pd.to_pickle(df, file_path)


def load_model(file_name: str, folder=None):
    """Load a model from a pickle file."""
    file_name_ext = file_name / '.pkl'
    if folder is None:
        file_path = MODELS_DIR / file_name_ext
    else:
        file_path = MODELS_DIR / folder / file_name_ext

    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model


def save_model(model, file_name: str, folder=None):
    """Save a model to a pickle file."""
    file_name_ext = file_name / '.pkl'
    if folder is None:
        file_path = MODELS_DIR / file_name_ext
    else:
        file_path = MODELS_DIR / folder / file_name_ext

    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
