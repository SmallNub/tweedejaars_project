import pandas as pd


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
