import pandas as pd
import numpy as np


def print_cond(msg, cond):
    """Print the message if the condition is true."""
    if cond:
        print(msg)


def get_submatrix(matrix, row_start=None, row_end=None, col_start=None, col_end=None, auto_flat=False):
    """Get a submatrix from a 2D list, handling empty slice arguments."""
    if row_start is None:
        row_start = 0
    if row_end is None:
        row_end = len(matrix)
    if col_start is None:
        col_start = 0
    if col_end is None:
        col_end = len(matrix[0])

    if auto_flat:
        if row_end - row_start == 1:
            if col_end - col_start == 1:
                return [row[col_start] for row in matrix[row_start]]
            else:
                return [row[col_start:col_end] for row in matrix[row_start]]
        elif col_end - col_start == 1:
            return [row[col_start] for row in matrix[row_start:row_end]]

    return [row[col_start:col_end] for row in matrix[row_start:row_end]]


def recast_pred(pred: np.ndarray):
    """Recast the predictions into a series object."""
    return pd.Series(pred, dtype=bool, name="pred")


def flatten_ptu(df: pd.DataFrame):
    """Flatten the PTU for metrics."""
    # Predefined possible operations
    base_agg_dict = {
        "target_two_sided_ptu": "any",
        "target_two_sided_ptu_alt": "any",
        "target_two_sided_ptu_realtime": "any",
        "target_two_sided_ptu_flip": "any",
        "pred": "any"
    }

    # Possible operations given the input
    agg_dict = {}
    for column in df.columns:
        if column in base_agg_dict:
            agg_dict[column] = base_agg_dict[column]

    if len(agg_dict) == 0:
        raise ValueError("Given DataFrame has no columns to flatten.")

    # Flatten
    flat_df = df.groupby("ptu_id").agg(agg_dict).reset_index()
    return flat_df
