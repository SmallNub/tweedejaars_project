import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from ..utility.flat import flatten_ptu


def show_basic_metrics(true, pred, ids, flatten=True):
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


def show_real_penalty_score(df: pd.DataFrame, true, pred, ids, example_revenue=False):
    """Calculates the penalty in revenue lost and gained."""
    df = df.copy()
    df['min_price'] = df['settlement_price_bestguess']
    df['max_price'] = df['settlement_price_bestguess']
    df['pred'] = pred
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
        'true': 'any'               # Is PTU two-sided
    }
    # Make it flat for easy calculations
    flat_df = df.groupby('id').aggregate(agg_dict)

    false_neg_penalty = flat_df['false_neg']
    false_pos_penalty = flat_df['false_pos']

    # Use the example revenue
    if example_revenue:
        energy = 100 / 60  # Example renewable energy
        false_neg_penalty *= flat_df['max_price'] * -energy
        false_pos_penalty *= flat_df['min_price'] * energy

    return false_neg_penalty.sum(), false_pos_penalty.sum()
