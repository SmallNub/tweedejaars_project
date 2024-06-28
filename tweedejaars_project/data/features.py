from pathlib import Path

import typer
import numpy as np
from loguru import logger
from tqdm import tqdm
import pandas as pd
from numba import jit

from tweedejaars_project.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MAIN_DATA_FILE_NAME
from tweedejaars_project.data.fileloader import load_df, save_df
from tweedejaars_project.evaluation.adjustment import detect_flip
from tweedejaars_project.utility.misc import lag

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

    # Use time since last two sided ptu to create the fixed target
    df[f"{output}_two_sided_ptu"] = df["time_since_last_two_sided"] == 0
    df[f"{output}_two_sided_ptu"] = df.groupby("ptu_id")[f"{output}_two_sided_ptu"].transform(set_first_two_false)

    logger.info(f"Added fix target. ({output}_two_sided_ptu)")
    return df


def add_fix_features(df: pd.DataFrame):
    """Adds multiple columns containing the fixed version of features."""
    def set_first_two_false(group):
        group.iloc[:2] = False
        return group

    transform_dict = {
        "fix_min_ptu_price_known": ["min_price_published", "cummin"],
        "fix_max_ptu_price_known": ["max_price_published", "cummax"],
    }
    # Fix min and max price known
    for fixed, [feature, operation] in transform_dict.items():
        df[fixed] = df.groupby("fix_ptu_id")[feature].transform(operation)
        df[fixed] = df.groupby("fix_ptu_id")[fixed].ffill()

    # Create the best guess using the fixed min and max price known
    df["fix_settlement_price_bestguess"] = df["fix_min_ptu_price_known"]
    mask = df["fix_max_ptu_price_known"].notna()
    df.loc[mask, "fix_settlement_price_bestguess"] = df.loc[mask, "fix_max_ptu_price_known"]

    # Add the alternative version of the fixed bestguess
    df = add_bestguess_alt(df, "fix_")

    # Create the fixed realized by taking the last value of the best guess
    df["fix_settlement_price_realized"] = df.groupby("fix_ptu_id")["fix_settlement_price_bestguess_alt"].transform("last")

    # Create the fixed versions of the targets using the new fixed features
    # Since "fix" version already existed before, this will receive "fix2"
    # "fix2" is not better than "fix", but is usefull for other feature creation
    df["fix2_two_sided_ptu_realtime"] = mask & df["fix_min_ptu_price_known"].notna()
    df["fix2_two_sided_ptu_alt"] = df.groupby("fix_ptu_id")["fix_two_sided_ptu_realtime"].transform("any")
    df["fix2_two_sided_ptu"] = df.groupby("fix_ptu_id")["fix_two_sided_ptu_alt"].transform(set_first_two_false)

    logger.info("Added several fixed features. (<many>)")
    return df


def add_binary_features(df: pd.DataFrame):
    """Adds a column containing binary features."""
    # The electricity is balanced if both min and max price are NaN
    df["is_balanced"] = df["min_price_published"].isna() & df["max_price_published"].isna()

    # The min price is currently negative
    df["down_negative"] = df["min_price_published"] < 0

    logger.info("Added binary features. (<many>)")
    return df


def add_time_since(df: pd.DataFrame, base="target", output=""):
    """Adds a column containing the time since last two sided ptu."""
    @jit
    def count_after(realtime: np.ndarray):
        """Counts the time after a two sided ptu."""
        out = np.empty_like(realtime, dtype=np.float_)
        count = 0
        add = 1 / 15  # time_since_last_two_sided counts is not in minutes but in PTUs
        for i in range(len(realtime)):
            # Is it currently a two sided ptu
            if realtime[i]:
                count = 0
            else:
                count += add
            out[i] = count
        return out

    df[f"{output}time_since_last_two_sided_alt"] = count_after(df[f"{base}_two_sided_ptu_realtime"].to_numpy())

    logger.info(f"Added alternative best guess. ({output}time_since_last_two_sided_alt)")
    return df


def add_bestguess_alt(df: pd.DataFrame, version=""):
    """Adds a column containing an alternative best guess by filling NaNs with mid price."""
    df[f"{version}settlement_price_bestguess_alt"] = df[f"{version}settlement_price_bestguess"]

    # Fill the NaN with mid price
    mask = df[f"{version}settlement_price_bestguess"].isna()
    df.loc[mask, f"{version}settlement_price_bestguess_alt"] = df.loc[mask, "mid_price_published"]

    logger.info(f"Added alternative best guess. ({version}settlement_price_bestguess_alt)")
    return df


def add_started_down(df: pd.DataFrame):
    """Adds a column containing if the ptu started with a down price."""
    df["started_down"] = df["min_price_published"].notna()
    df["started_down"] = df.groupby("fix_ptu_id")["started_down"].transform("first")

    logger.info("Added indicator for start on down. (started_down)")
    return df


def add_residual_load(df: pd.DataFrame):
    """Adds a column containing the residual load."""
    df["residual_load"] = df["forecast_demand"] - df["forecast_solar"] - df["forecast_wind"]

    logger.info("Added residual load. (residual_load)")
    return df


def add_forecast_deltas(df: pd.DataFrame):
    """Adds multiple columns containing the deltas of the forecasts."""
    df["forecast_wind_delta"] = df["forecast_wind"].diff(15)
    df["forecast_solar_delta"] = df["forecast_solar"].diff(15)
    df["forecast_demand_delta"] = df["forecast_demand"].diff(15)

    logger.info("Added deltas of forecasts. (<many>)")
    return df


def add_price_volume(df: pd.DataFrame):
    """Adds multiple columns containing price volume features."""
    df["down_price_volume"] = df["downward_dispatch_published"] * df["min_price_published"]
    df["up_price_volume"] = df["upward_dispatch_published"] * df["max_price_published"]
    df["diff_price_volume"] = df["up_price_volume"] - df["down_price_volume"]

    logger.info("Added price volume features. (<many>)")
    return df


def add_eneco_features(df: pd.DataFrame):
    """Adds multiple columns containing features from Eneco."""
    df["max_price_filled_known"] = df["max_ptu_price_known"]
    mask = df["max_ptu_price_known"].isna()
    df.loc[mask, "max_price_filled_known"] = df.loc[mask, "mid_price_published"]
    df["last_15min_up_volume"] = df["upward_dispatch_published"].rolling(15, 1).sum()
    df["last_15min_down_volume"] = df["downward_dispatch_published"].rolling(15, 1).sum()

    logger.info("Added Eneco features. (<many>)")
    return df


def add_diff_features(df: pd.DataFrame):
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


def add_peak_features(df: pd.DataFrame):
    """Adds multiple columns containing peak features."""
    def peak_features(df: pd.DataFrame, feature: str, negative=False):
        """Create the peak features."""
        shifted_1 = df[feature].shift(1)
        shifted_2 = df[feature].shift(2)
        if negative:
            df[f"{feature}_peak"] = (df[feature] >= shifted_1) & (shifted_1 < shifted_2)
        else:
            df[f"{feature}_peak"] = (df[feature] <= shifted_1) & (shifted_1 > shifted_2)

        df[f"{feature}_peak_time"] = df[f"{feature}_peak"].cumsum()
        df[f"{feature}_peak_time"] = df.groupby(f"{feature}_peak_time").cumcount() + 1

        peak_values = pd.Series(np.where(df[f"{feature}_peak"], df[feature].shift(1), np.nan)).ffill()
        df[f"{feature}_peak_diff"] = df[feature] - peak_values

        return df

    features = {
        "downward_dispatch_published": [],
        "upward_dispatch_published": [],
        "igcc_contribution_down_published": [],
        "igcc_contribution_up_published": [],
        "min_price_published": [True],
        "max_price_published": [],
    }

    for feature, args in features.items():
        df = peak_features(df, feature, *args)

    logger.info("Added several peak features. (<many>)")
    return df


def add_lagged_features(df: pd.DataFrame):
    """Adds multiple columns containing lagged features."""
    features = {
        "min_price_published": [1],
        "max_price_published": [1],
        "fix_two_sided_ptu": [17],
        "settlement_price_realized": [17],
        "minute_in_ptu": [2]
    }
    for feature, amounts in features.items():
        for amount in amounts:
            df = lag(df, feature, amount)

    logger.info("Added several lagged features. (<many>)")
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
        (add_fix_features, (), {}),
        (add_binary_features, (), {}),
        (add_time_since, (), {"base": "fix2"}),
        (add_bestguess_alt, (), {}),
        (add_started_down, (), {}),
        (add_residual_load, (), {}),
        (add_forecast_deltas, (), {}),
        (add_price_volume, (), {}),
        (add_eneco_features, (), {}),
        (add_diff_features, (), {}),
        (add_time_features, (), {}),
        (add_peak_features, (), {}),
        (add_lagged_features, (), {})
    ]

    logger.info("Generating features from dataset...")
    for func, args, kwargs in tqdm(tasks, desc="Generating features"):
        df = func(df, *args, **kwargs)

    df = save_df(df, output_path)
    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()
