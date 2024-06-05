import numpy as np
import pandas as pd

# Standard percentage of data to be used for training, validating and testing
VALID_PERCENTAGE = 0.25
TEST_PERCENTAGE = 0.25

# Target variable for training and testing
TARGET = 'target_two_sided_ptu'

REQUIRED_COLUMNS = ['datetime', 'ptu_id', TARGET]

# Features to be used for training and testing
FEATURES = [

]


def get_splits(
    df: pd.DataFrame,
    features: list[str] = FEATURES,
    target=TARGET,
    valid_percentage=VALID_PERCENTAGE,
    test_percentage=TEST_PERCENTAGE
) -> dict[list]:
    """
    Split the data into training, validation and test sets for a single target variable.

    Returns:
        dict: A dictionary containing training, validation and test sets for features and targets.
    """
    data = df[features + REQUIRED_COLUMNS]
    test_date = get_split_date(data, 1 - TEST_PERCENTAGE)
    train_val = 

    # Prepare train data
    train_features = train.drop([target] + NEEDED_COLUMNS, axis=1).reset_index(drop=True)
    train_target = train[target].reset_index(drop=True)
    train_ids = train["job_id"].reset_index(drop=True)

    # Prepare test data
    test_features = test.drop([target] + NEEDED_COLUMNS, axis=1).reset_index(drop=True)
    test_target = test[target].reset_index(drop=True)
    test_ids = test["job_id"].reset_index(drop=True)

    # Store split data
    split_data = {
        "train": [train_features, train_target, train_ids],
        "test": [test_features, test_target, test_ids]
    }

    return split_data


def get_split_date(df: pd.DataFrame, split_percentage: float):
    """Calculate the split date based on the given percentage of data."""
    idx = np.round((len(df['datetime']) * split_percentage))
    date = df.loc[idx, 'datetime'].dt.round('D')
    return date
