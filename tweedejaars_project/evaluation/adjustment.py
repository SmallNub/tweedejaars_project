import pandas as pd
from numba import jit


@jit
def change_after_flip(target, value):
    """Change the values after flipping."""
    length = len(target)
    for i in range(0, length, 15):
        end = min(i + 15, length)
        for idx in range(i, end):
            if target[idx]:
                target[idx + 1:end] = value
    return target


def detect_flip(target: pd.Series):
    """Sets the flipping point to True, the rest to False. Works best with adjusted target."""
    return pd.Series(change_after_flip(target.to_numpy(copy=True), False), name="flip")


def adjust_pred_consistency(pred: pd.Series):
    """
    Adjust the predictions to be consistent.\\
    If it predicts True earlier, it will keep being True for the rest of the PTU.
    """
    return pd.Series(change_after_flip(pred.to_numpy(copy=True), True), name="pred")


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
    return pd.Series(set_first_two_false(pred.to_numpy(copy=True)), name="pred")
