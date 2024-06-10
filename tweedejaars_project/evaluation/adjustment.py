import pandas as pd


def detect_flip(realtime: pd.Series, target: pd.Series):
    """Detect when the PTU flips to two-sided. Works best with adjusted target."""
    return target.diff() & realtime


def adjust_pred_consistency(pred: pd.Series, ids: pd.Series):
    """
    Adjust the predictions to be consistent.\\
    If it predicts True earlier, it will keep being True for the rest of the PTU.
    """
    df = pd.concat([pred, ids], axis=1)
    df.columns = ["pred", "id"]
    df["pred"] = df.groupby("id")["pred"].transform("any")
    return df["pred"]


def adjust_pred_realtime(realtime: pd.Series, pred: pd.Series):
    """
    Adjust the predictions to be at least as good as realtime.
    It will merge the predictions and the realtime target.
    """
    return realtime | pred
