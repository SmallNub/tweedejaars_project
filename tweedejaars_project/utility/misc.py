import pandas as pd


def flatten_ptu(df: pd.DataFrame):
    """Flatten the PTU for metrics."""
    # Predefined possible operations
    base_agg_dict = {
        'target_two_sided_ptu': 'any',
        'target_two_sided_ptu_alt': 'any',
        'target_two_sided_ptu_realtime': 'any',
        'target_two_sided_ptu_flip': 'any',
        'pred': 'any'
    }

    # Possible operations given the input
    agg_dict = {}
    for column in df.columns:
        if column in base_agg_dict:
            agg_dict[column] = base_agg_dict[column]

    if len(agg_dict) == 0:
        raise ValueError('Given DataFrame has no columns to flatten.')

    # Flatten
    flat_df = df.groupby("ptu_id").agg(agg_dict).reset_index()
    return flat_df


def realtime_adjustment(df, pred):
    """Adjust the predictions to be at least as good as realtime"""
    return df['target_two_sided_ptu_realtime'] | pred


def detect_flip(df: pd.DataFrame, target: pd.Series, adjust=True):
    """Detect when the PTU flips to two-sided. Works best with adjusted target."""
    if adjust:
        target = realtime_adjustment(df, target)

    return target.diff() & df['target_two_sided_ptu_realtime']
