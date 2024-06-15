import pandas as pd
from numba import jit


def detect_flip(target: pd.Series):
    """Detect when the PTU flips to two-sided. Works best with adjusted target."""
    @jit
    def get_flip(target):
        length = len(target)
        for i in range(0, length, 15):
            end = min(i + 15, length)
            for idx in range(i, end):
                if target[idx]:
                    target[idx + 1:end] = False
        return target

    return pd.Series(get_flip(target.to_numpy()), name="flip")


def adjust_pred_consistency(pred: pd.Series, ids: pd.Series):
    """
    Adjust the predictions to be consistent.\\
    If it predicts True earlier, it will keep being True for the rest of the PTU.
    """
    df = pd.concat([pred, ids], axis=1)
    df.columns = ["pred", "id"]
    df["pred"] = df.groupby("id")["pred"].transform("cummax", engine="numba", engine_kwargs={"parallel": True})
    return df["pred"]


def adjust_pred_realtime(realtime: pd.Series, pred: pd.Series):
    """
    Adjust the predictions to be at least as good as realtime.
    It will merge the predictions and the realtime target.
    """
    return realtime | pred


def adjust_pred_conform(pred: pd.Series):
    """Set the first values in each PTU to be false to correspond to the original target."""
    @jit
    def set_first_two_false(pred):
        for i in range(0, len(pred), 15):
            pred[i] = False
            pred[i + 1] = False
        return pred
    return pd.Series(set_first_two_false(pred.to_numpy()), name="pred")
