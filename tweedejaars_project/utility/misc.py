import pandas as pd
import numpy as np
import time
import functools


def time_func(func):
    """Print the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # High-resolution start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.perf_counter()  # High-resolution end time
        execution_time = end_time - start_time  # Calculate the execution time
        print(f"Function '{func.__name__}' executed in {execution_time:.8f} seconds")
        return result  # Return the result of the function
    return wrapper


def print_cond(msg, cond):
    """Print the message if the condition is true."""
    if cond:
        print(msg)


def get_submatrix(matrix, row_start=None, row_end=None, col_start=None, col_end=None, auto_flat=False):
    """Get a submatrix from a 2D nested list, handling empty slice arguments."""
    # Autofill omitted values
    if row_start is None:
        row_start = 0
    if row_end is None:
        row_end = len(matrix)
    if col_start is None:
        col_start = 0
    if col_end is None:
        col_end = len(matrix[0])

    # If the matrix becomes a shape of (n, 1) or (1, n), flatten it to be (n) or (1)
    if auto_flat:
        # Check if only 1 row is selected
        if row_end - row_start == 1:
            # Check if only 1 column is selected
            if col_end - col_start == 1:
                # Only 1 item, since 1 row and 1 column
                return matrix[row_start][col_start]
            else:
                # Subset of the selected row
                return [row[col_start:col_end] for row in matrix[row_start]]
        # Check if only 1 column is selected
        elif col_end - col_start == 1:
            # Subset of the selected column
            return [row[col_start] for row in matrix[row_start:row_end]]

    # Default behaviour
    return [row[col_start:col_end] for row in matrix[row_start:row_end]]


def transpose_matrix(matrix):
    """Get the transpose of a 2D nested list."""
    out = [[] for _ in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            out[j].append(matrix[i][j])
    return out


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


def lag(df: pd.DataFrame, feature: str, amount=1):
    """Lag a feature by an amount."""
    lagged_feature = f"{feature}_{amount}"
    df[lagged_feature] = df[feature].shift(amount)
    return df
