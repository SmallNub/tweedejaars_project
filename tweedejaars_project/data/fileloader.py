from pathlib import Path
import pandas as pd

from ..config import PROCESSED_DATA_DIR, MAIN_DATA_FILE_NAME


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
