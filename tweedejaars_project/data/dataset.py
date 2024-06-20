from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd

from tweedejaars_project.config import INTERIM_DATA_DIR, RAW_DATA_DIR, MAIN_DATA_FILE_NAME
from tweedejaars_project.data.fileloader import load_df, save_df
from tweedejaars_project.data.features import add_ids, add_realtime_target, add_fix_target

app = typer.Typer()


def fix_target(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the target to be more correct and consistent."""
    temp_df = pd.concat([df["datetime"], df["time_since_last_two_sided"]], axis=1)
    temp_df.columns = ["datetime", "time_since_last_two_sided"]
    temp_df = add_ids(temp_df)
    temp_df = add_realtime_target(temp_df)
    temp_df = add_fix_target(temp_df)
    df["target_two_sided_ptu"] = temp_df["fix_two_sided_ptu_realtime"]
    logger.info("Fixed the target.")
    return df


def fix_nan(df: pd.DataFrame):
    """Forward fills some features to fix NaNs."""
    features = [
        "upward_dispatch_published",
        "downward_dispatch_published",
        "igcc_contribution_up_published",
        "igcc_contribution_down_published",
        "forecast_wind",
        "forecast_solar",
        "forecast_demand",
    ]
    df[features] = df[features].ffill()
    logger.info("Fixed all the NaNs in some features.")
    return df


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / MAIN_DATA_FILE_NAME,
    output_path: Path = INTERIM_DATA_DIR / MAIN_DATA_FILE_NAME,
):
    logger.info("Loading dataset and initializing operations...")
    df = load_df(input_path)

    # List of (feature_function, args, kwargs) tuples
    tasks = [
        (fix_nan, (), {}),
    ]

    logger.info("Processing dataset...")
    for func, args, kwargs in tqdm(tasks, desc="Processing data"):
        df = func(df, *args, **kwargs)

    df = save_df(df, output_path)
    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
