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


def add_alt_target(df: pd.DataFrame):
    """Adds a column containing an alternative target, the first two minutes are also counted."""
    df['target_two_sided_ptu_alt'] = df.groupby('ptu_id')['target_two_sided_ptu'].transform('any')
    return df


def add_realtime_target(df: pd.DataFrame):
    """Adds a column containing a real-time version of the target."""
    df['target_two_sided_ptu_realtime'] = df['time_since_last_two_sided'] == 0
    return df


def add_flip_target(df: pd.DataFrame):
    """Adds a column containing when the ptu flipped to two-sided."""
    df['target_two_sided_ptu_flip'] = df['target_two_sided_ptu_realtime'].diff() & df['target_two_sided_ptu_realtime']
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
            logger.info('Added id for each ptu. (ptu_id)')
        if i == 1:
            df = add_alt_target(df)
            logger.info('Added alternative target. (target_two_sided_ptu_alt)')
        if i == 2:
            df = add_realtime_target(df)
            logger.info('Added realtime target. (target_two_sided_ptu_realtime)')
        if i == 3:
            df = add_flip_target(df)
            logger.info('Added flip target. (target_two_sided_ptu_flip)')

    df = save_df(df, output_path)
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
