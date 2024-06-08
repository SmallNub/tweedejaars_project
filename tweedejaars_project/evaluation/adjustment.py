import pandas as pd


def realtime_adjustment(df, pred):
    """Adjust the predictions to be at least as good as realtime"""
    return df['target_two_sided_ptu_realtime'] | pred
