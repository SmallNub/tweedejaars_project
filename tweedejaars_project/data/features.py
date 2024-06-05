from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd

from ..config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MAIN_DATA_FILE_NAME
from .fileloader import load_df, save_df

app = typer.Typer()


def add_ids(df: pd.DataFrame):
    """Adds a column containing an unique id for each ptu."""
    df['ptu_id'] = (df['datetime'] - df['datetime'].min()) // pd.Timedelta(minutes=15)
    return df


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = INTERIM_DATA_DIR / MAIN_DATA_FILE_NAME,
    output_path: Path = PROCESSED_DATA_DIR / MAIN_DATA_FILE_NAME,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    df = load_df(input_path)

    for i in tqdm(range(10), total=10):
        if i == 0:
            df = add_ids(df)
        if i == 5:
            logger.info("Something happened for iteration 5.")

    df = save_df(df, output_path)
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
