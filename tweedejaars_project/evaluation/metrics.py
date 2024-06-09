import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from ..utility import flatten_ptu, detect_flip


def show_basic_metrics(true: pd.Series, pred: pd.Series, ids: pd.Series, flatten=True):
    """Show basic metric performance, classification report and confusion matrix."""
    if flatten:
        # Concatenate the true, pred, and ids into a single DataFrame
        base = pd.concat([pd.Series(true), pd.Series(pred), pd.Series(ids)], axis=1, ignore_index=True)
        base.columns = ["target_two_sided_ptu", "pred", "ptu_id"]
        base = flatten_ptu(base)
        true = base['target_two_sided_ptu']
        pred = base['pred']

    print("Classification Report:")
    print(classification_report(true, pred))

    print("Confusion Matrix:")
    ConfusionMatrixDisplay.from_predictions(true, pred)
    plt.show()


def show_real_penalty_score(df: pd.DataFrame, true: pd.Series, pred: pd.Series, ids: pd.Series, example_revenue=False):
    """Calculates the penalty in revenue lost and gained."""
    df = df.copy()
    df['min_price'] = df['min_price_published']
    df['max_price'] = df['max_price_published']
    df['pred'] = pd.Series(pred, dtype=bool)
    df['id'] = ids
    df['true'] = true

    # False negative with respect to naive strategy action
    # If it is a two-sided PTU and it curtails, it will be a false negative if the prediction is false
    df['false_neg'] = False
    df['has_impact_neg'] = df['true'] & df['naive_strategy_action']
    df.loc[df['has_impact_neg'], 'false_neg'] = ~df.loc[df['has_impact_neg'], 'pred']

    # False positive with respect to naive strategy action
    # If it is a one-sided PTU and it curtails, it will be a false positive if the prediction is true
    df['false_pos'] = False
    df['has_impact_pos'] = ~df['true'] & df['naive_strategy_action']
    df.loc[df['has_impact_pos'], 'false_pos'] = df.loc[df['has_impact_pos'], 'pred']

    agg_dict = {
        'min_price': 'min',         # Min down price
        'max_price': 'max',         # Max up price
        'has_impact_neg': 'sum',    # Count total possible false negatives
        'has_impact_pos': 'sum',    # Count total possible false positives
        'false_neg': 'sum',         # Count all the false negatives
        'false_pos': 'sum',         # Count all the false positives
    }
    # Make it flat for easy calculations
    flat_df = df.groupby('id').agg(agg_dict)

    false_neg_penalty = flat_df['false_neg']
    false_pos_penalty = flat_df['false_pos']
    false_neg_penalty_total = flat_df['has_impact_neg']
    false_pos_penalty_total = flat_df['has_impact_pos']

    # Use the example revenue
    if example_revenue:
        energy = 100 / 60  # Example renewable energy
        false_neg_penalty *= flat_df['max_price'] * -energy
        false_pos_penalty *= flat_df['min_price'] * energy
        false_neg_penalty_total *= flat_df['max_price'] * -energy
        false_pos_penalty_total *= flat_df['min_price'] * energy

    false_neg_penalty_sum = false_neg_penalty.sum()
    false_pos_penalty_sum = false_pos_penalty.sum()
    false_neg_penalty_total_sum = false_neg_penalty_total.sum()
    false_pos_penalty_total_sum = false_pos_penalty_total.sum()

    # Invert it so it makes more logical sense
    # false_neg_penalty_sum = false_neg_penalty_total_sum - false_neg_penalty_sum
    # false_pos_penalty_sum = false_pos_penalty_total_sum - false_pos_penalty_sum

    print(f"False negative score (pred/max): {false_neg_penalty_sum / false_neg_penalty_total_sum}, {false_neg_penalty_sum}/{false_neg_penalty_total_sum}")
    print(f"False positive score (pred/max): {false_pos_penalty_sum / false_pos_penalty_total_sum}, {false_pos_penalty_sum}/{false_pos_penalty_total_sum}")
    # return false_neg_penalty_sum, false_neg_penalty_total_sum, false_pos_penalty_sum, false_pos_penalty_total_sum

# TODO make it work better with adjustment
def show_time_diff_score(df: pd.DataFrame, pred: pd.Series, ids: pd.Series):
    df = df.copy()
    df['start_idx'] = True
    df['true'] = df['target_two_sided_ptu_realtime']
    df['pred'] = pd.Series(pred, dtype=bool)
    df['id'] = ids
    df['flip'] = detect_flip(df, df['pred'], False)

    agg_dict = {
        'start_idx': 'idxmax',                      # Get the start index of the PTU
        'flip': 'idxmax',                           # The time it flips
        'target_two_sided_ptu_realtime': 'idxmax',  # The time it has to flip
        'true': 'any',                              # Is the PTU two-sided
        'pred': 'any'                               # Prediction PTU two-sided
    }
    flat_df = df.groupby('id').agg(agg_dict)

    # True positives
    true_pos_mask = flat_df['true'] & flat_df['pred']
    true_pos_time = flat_df.loc[true_pos_mask, 'flip'] - flat_df.loc[true_pos_mask, 'target_two_sided_ptu_realtime']
    true_pos_time_avg = true_pos_time.mean()
    true_pos_count = true_pos_mask.sum()

    # Split into positive and negative time differences
    true_pos_time_mask = true_pos_time < 0
    true_pos_time_neg = true_pos_time[true_pos_time_mask].describe()
    true_pos_time_pos = true_pos_time[~true_pos_time_mask].describe()

    # Best case scenario
    true_time = flat_df.loc[flat_df['true'], 'start_idx'] - flat_df.loc[flat_df['true'], 'target_two_sided_ptu_realtime']
    true_time_avg = true_time.mean()
    true_count = flat_df['true'].sum()

    time_df = pd.concat([true_pos_time_neg, true_pos_time_pos], axis=1)
    time_df.columns = ['neg', 'pos']
    print(time_df)
    print(f"Time taken (pred/max): {true_pos_time_avg}/{true_time_avg}, using {true_pos_count}/{true_count}")
    # return true_pos_time_neg, true_pos_time_pos, true_pos_time_avg, true_pos_count, true_time_avg, true_count
