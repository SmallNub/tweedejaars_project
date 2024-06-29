import numpy as np
import pandas as pd

# Standard percentage of data to be used for training, validating and testing
VALID_PERCENTAGE = 0.25
TEST_PERCENTAGE = 0.25

# Default target
TARGET = "target_two_sided_ptu"

# Default features
FEATURES = [

]


def get_splits(
    df: pd.DataFrame,
    features: list[str] = FEATURES,
    target=TARGET,
    valid_percentage=VALID_PERCENTAGE,
    test_percentage=TEST_PERCENTAGE,
    return_dict=True,
    return_dict_pair=True,
) -> dict[list] | list[list]:
    """
    Split the data into training, validation and test sets for a single target variable.

    Returns:
        dict: A dictionary containing training, validation and test sets for features and targets.
    """
    df = df.copy()

    # Test set
    test_date = get_split_date(df, 1 - test_percentage)
    train_valid, test = split_on_date(df, test_date)

    # Training and validation set
    valid_date = get_split_date(train_valid, 1 - valid_percentage)
    train, valid = split_on_date(train_valid, valid_date)

    # Get the pairs for each set
    splits = {
        "train": split_into_pair(train, features, target, return_dict_pair),
        "valid": split_into_pair(valid, features, target, return_dict_pair),
        "test": split_into_pair(test, features, target, return_dict_pair),
        "train_valid": split_into_pair(train_valid, features, target, return_dict_pair),
        "full": split_into_pair(df, features, target, return_dict_pair),
    }

    if return_dict:
        return splits

    return list(splits.values())


def get_split_date(df: pd.DataFrame, split_percentage: float):
    """Calculate the split date using the percentage, rounded to the nearest day."""
    idx = np.round((len(df["datetime"]) * split_percentage))
    date = df.loc[idx, "datetime"].round("D")
    return date


def split_on_date(df: pd.DataFrame, split_date: str):
    """Split the data using the split date."""
    split_date = pd.to_datetime(split_date)
    first = df[df["datetime"] < split_date].reset_index(drop=True)
    second = df[df["datetime"] >= split_date].reset_index(drop=True)
    return [first, second]


def split_into_pair(df: pd.DataFrame, features: list[str], target: str, return_dict=True):
    """Split the data into pairs of input features and output targets."""
    in_pair = df[features].reset_index(drop=True)
    out_pair = df[target].reset_index(drop=True)
    id_pair = df["ptu_id"].reset_index(drop=True)
    df_pair = df.reset_index(drop=True)

    if return_dict:
        return {"in": in_pair, "out": out_pair, "id": id_pair, "df": df_pair}

    return [in_pair, out_pair, id_pair, df_pair]
