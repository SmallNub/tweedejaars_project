import pandas as pd

from ..evaluation.adjustment import realtime_adjustment


def flatten_ptu(df: pd.DataFrame):
    """Flatten the PTU for metrics."""
    # Predefined possible operations
    base_agg_dict = {
        'target_two_sided_ptu': 'any',
        'pred': 'any'
    }

    # Possible operations given the input
    agg_dict = {}
    for column in df.columns:
        if column in base_agg_dict:
            agg_dict = base_agg_dict[column]

    if len(agg_dict) == 0:
        raise ValueError('Given DataFrame has no columns to flatten.')

    # Flatten
    df = df.groupby("ptu_id").agg(agg_dict).reset_index()
    return df


def detect_flip(df: pd.DataFrame, target: pd.Series, adjust=True):
    """Detect when the PTU flips to two-sided."""
    if adjust:
        target = realtime_adjustment(df, target)

    df['target_two_sided_ptu_flip'] = target.diff() & df['target_two_sided_ptu_realtime']
    return df
