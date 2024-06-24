from pathlib import Path

import typer
import numpy as np
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

    df["fix_ptu_id"] = df["ptu_id"].shift(2, fill_value=-1)
    logger.info("Added id for each ptu with delay. (fix_ptu_id)")
    return df


def add_alt_target(df: pd.DataFrame, version="target"):
    """Adds a column containing an alternative target, the first two minutes are also counted."""
    df[f"{version}_two_sided_ptu_alt"] = df.groupby("fix_ptu_id")[f"{version}_two_sided_ptu"].transform("any")
    logger.info(f"Added alternative target. ({version}_two_sided_ptu_alt)")
    return df


def add_realtime_target(df: pd.DataFrame, version="target"):
    """Adds a column containing a real-time version of the target."""
    df[f"{version}_two_sided_ptu_realtime"] = (df["time_since_last_two_sided"] == 0) & df[f"{version}_two_sided_ptu_alt"]
    logger.info(f"Added realtime target. ({version}_two_sided_ptu_realtime)")
    return df


def add_flip_target(df: pd.DataFrame, version="target"):
    """Adds a column containing when the ptu flipped to two-sided."""
    df[f"{version}_two_sided_ptu_flip"] = detect_flip(df[f"{version}_two_sided_ptu_realtime"])
    logger.info(f"Added flip target. ({version}_two_sided_ptu_flip)")
    return df


def add_fix_target(df: pd.DataFrame, output="fix"):
    """Adds a column containing the fixed version of the target."""
    def set_first_two_false(group):
        group.iloc[:] = group.any()
        group.iloc[:2] = False
        return group

    df[f"{output}_two_sided_ptu"] = df["time_since_last_two_sided"] == 0
    df[f"{output}_two_sided_ptu"] = df.groupby("ptu_id")[f"{output}_two_sided_ptu"].transform(set_first_two_false)
    logger.info(f"Added fix target. ({output}_two_sided_ptu)")
    return df


def add_binary_features(df: pd.DataFrame):
    """Adds a column containing binary features."""
    df["is_balanced"] = df["min_price_published"].isna() & df["max_price_published"].isna()
    df["down_negative"] = df["min_price_published"] < 0
    logger.info("Added binary features. (<many>)")
    return df


def add_fix_action(df: pd.DataFrame, version="target"):
    df["down_negative"] & ~df[f"{version}_two_sided_ptu_alt"]


def add_residual_load(df: pd.DataFrame):
    """Adds a column containing the residual load."""
    df["residual_load"] = df["forecast_demand"] - df["forecast_solar"] - df["forecast_wind"]
    logger.info("Added residual load. (residual_load)")
    return df


def add_forecast_deltas(df: pd.DataFrame):
    """Adds multiple columns containing the deltas of the forcasts."""
    df["forecast_wind_delta"] = df["forecast_wind"].diff(15)
    df["forecast_solar_delta"] = df["forecast_solar"].diff(15)
    df["forecast_demand_delta"] = df["forecast_demand"].diff(15)
    logger.info("Added deltas of forcasts. (<many>)")
    return df


def add_price_volume(df: pd.DataFrame):
    """Adds multiple columns containing price volume features. (NOTE decreases income)"""
    df["down_price_volume"] = df["downward_dispatch_published"] * df["min_price_published"]
    df["up_price_volume"] = df["upward_dispatch_published"] * df["max_price_published"]
    df["diff_price_volume"] = df["up_price_volume"] - df["down_price_volume"]
    logger.info("Added price volume features. (<many>)")
    return df


def add_diff(df: pd.DataFrame):
    """Adds multiple columns containing differences of related columns."""
    df["dispatch_diff"] = df["upward_dispatch_published"] - df["downward_dispatch_published"]
    df["igcc_diff"] = df["igcc_contribution_up_published"] - df["igcc_contribution_down_published"]
    logger.info("Added difference of related columns. (<many>)")
    return df


def add_time_features(df: pd.DataFrame):
    """Adds multiple columns containing time related features."""
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["weekday"] = df["datetime"].dt.weekday
    df["weekday_ptu"] = df["weekday"] * 96 + df["PTU"]
    df["weekday_hour"] = df["weekday"] * 24 + df["hour"]
    df["workday"] = df["weekday"].isin(range(5))
    logger.info("Added several time features. (<many>)")
    return df

def peak_features(df,feature,negative=False):
    if negative:
        df[f'{feature}_peak'] = (df[feature] >= df[feature].shift(1)) & (df[feature].shift(1) < df[feature].shift(2))
    else:
        df[f'{feature}_peak'] = (df[feature] <= df[feature].shift(1)) & (df[feature].shift(1) > df[feature].shift(2))

    df[f'{feature}_peak_time'] = df[f'{feature}_peak'].cumsum()
    df[f'{feature}_peak_time'] = df.groupby(f'{feature}_peak_time').cumcount() + 1

    df['peak_values'] = np.where(df[f'{feature}_peak'], df[feature].shift(1), np.nan)
    df['peak_values'] = df['peak_values'].ffill()
    df[f'{feature}_peak_diff'] = df[feature] - df['peak_values'].ffill()

def add_peak_features(df: pd.DataFrame):
    peak_features(df,'downward_dispatch_published')
    peak_features(df,'upward_dispatch_published')
    peak_features(df,'igcc_contribution_down_published')
    peak_features(df,'igcc_contribution_up_published')
    peak_features(df,'min_price_published',negative=True)
    peak_features(df,'max_price_published')
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
        (add_binary_features, (), {}),
        (add_residual_load, (), {}),
        (add_forecast_deltas, (), {}),
        (add_price_volume, (), {}),
        (add_diff, (), {}),
        (add_time_features, (), {}),
        (add_peak_features, (), {})
    ]

    logger.info("Generating features from dataset...")
    for func, args, kwargs in tqdm(tasks, desc="Generating features"):
        df = func(df, *args, **kwargs)

    df = save_df(df, output_path)
    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()
