from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd

from tweedejaars_project.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MAIN_DATA_FILE_NAME
from tweedejaars_project.data.fileloader import load_df, save_df
from tweedejaars_project.evaluation.adjustment import detect_flip

app = typer.Typer()


def add_ids(df: pd.DataFrame):
    """Adds a column containing an unique id for each ptu."""
    df["ptu_id"] = (df["datetime"] - df["datetime"].min()) // pd.Timedelta(minutes=15)
    logger.info("Added id for each ptu. (ptu_id)")
    return df


def add_alt_target(df: pd.DataFrame, version="target"):
    """Adds a column containing an alternative target, the first two minutes are also counted."""
    df[f"{version}_two_sided_ptu_alt"] = df.groupby("ptu_id")[f"{version}_two_sided_ptu"].transform("any")
    logger.info(f"Added alternative target. ({version}_two_sided_ptu_alt)")
    return df


def add_realtime_target(df: pd.DataFrame, version="target"):
    """Adds a column containing a real-time version of the target."""
    df[f"{version}_two_sided_ptu_realtime"] = (df["time_since_last_two_sided"] == 0) & df[f"{version}_two_sided_ptu_alt"]
    logger.info(f"Added realtime target. ({version}_two_sided_ptu_realtime)")
    return df


def add_flip_target(df: pd.DataFrame, version="target"):
    """Adds a column containing when the ptu flipped to two-sided."""
    df[f"{version}_two_sided_ptu_flip"] = detect_flip(df[f"{version}_two_sided_ptu_realtime"], df[f"{version}_two_sided_ptu_realtime"])
    logger.info(f"Added flip target. ({version}_two_sided_ptu_flip)")
    return df


def add_fix_target(df: pd.DataFrame, base="target", output="fix"):
    """Adds a column containing the fixed version of the target."""
    def set_first_two_false(group):
        group.iloc[:] = group.any()
        group.iloc[:2] = False
        return group

    df[f"{output}_two_sided_ptu"] = df.groupby("ptu_id")[f"{base}_two_sided_ptu_realtime"].transform(set_first_two_false)
    return df


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / MAIN_DATA_FILE_NAME,
    output_path: Path = PROCESSED_DATA_DIR / MAIN_DATA_FILE_NAME,
):
    logger.info("Loading dataset and initializing operations...")
    df = load_df(input_path)

    # List of (feature_function, args, kwargs) tuples
    tasks = [
        (add_ids, (), {}),
        (add_alt_target, (), {}),
        (add_realtime_target, (), {}),
        (add_flip_target, (), {}),
        (add_fix_target, (), {}),
        (add_alt_target, (), {"version": "fix"}),
        (add_realtime_target, (), {"version": "fix"}),
        (add_flip_target, (), {"version": "fix"}),
    ]

    logger.info("Generating features from dataset...")
    for func, args, kwargs in tqdm(tasks, desc="Generating features"):
        df = func(df, *args, **kwargs)

    df = save_df(df, output_path)
    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()
