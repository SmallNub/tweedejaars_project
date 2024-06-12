import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from tweedejaars_project.utility import flatten_ptu, get_submatrix
from tweedejaars_project.evaluation import detect_flip, adjust_pred_realtime, adjust_pred_consistency
from tweedejaars_project.visualization import default_titles, make_subplots, \
    plot_classification_report, plot_confusion_matrix, plot_penalty_score, plot_time_diff_df, plot_time_diff_avg


def compute_basic_metrics(true: pd.Series, pred: pd.Series, ids: pd.Series, flatten=True):
    """Compute basic metric performance, classification report and confusion matrix."""
    if flatten:
        # Concatenate the true, pred, and ids into a single DataFrame
        base = pd.concat([true, pred, ids], axis=1)
        base.columns = ["target_two_sided_ptu", "pred", "ptu_id"]
        base = flatten_ptu(base)
        true = base["target_two_sided_ptu"]
        pred = base["pred"]

    report = classification_report(true, pred, output_dict=True)
    matrix = confusion_matrix(true, pred)
    return report, matrix, flatten


def print_basic_metrics(results, titles=None):
    """Print the basic metrics."""
    titles = default_titles(results, titles)

    # Check if there are multiple results
    try:
        flat = results[0][2]
    except KeyError:
        flat = results[2]

    # Create subplots for classification reports
    make_subplots(
        lambda result, title, ax: plot_classification_report(result[0], title, ax),
        results,
        titles,
        suptitle=f"Comparison of Classification Reports - {"Flattened per PTU" if flat else "Default"}",
        figsize=(18, 2.5)
    )

    # Create subplots for confusion matrices
    make_subplots(
        lambda result, title, ax: plot_confusion_matrix(result[1], title, ax),
        results,
        titles,
        suptitle=f"Comparison of Confusion Matrices - {"Flattened per PTU" if flat else "Default"}",
        figsize=(18, 5)
    )


def show_basic_metrics(true: pd.Series, pred: pd.Series, ids: pd.Series, flatten=True):
    """Show basic metric performance, classification report and confusion matrix."""
    results = compute_basic_metrics(true, pred, ids, flatten)
    print_basic_metrics(results)


def compute_penalty_score(df: pd.DataFrame, true: pd.Series, pred: pd.Series, ids: pd.Series, example_revenue=False):
    """Calculates the penalty in revenue lost and gained."""
    df = df.copy()
    df["min_price"] = df["min_price_published"]
    df["max_price"] = df["max_price_published"]
    df["pred"] = pred
    df["id"] = ids
    df["true"] = true

    # False negative with respect to naive strategy action
    # If it is a two-sided PTU and it curtails, it will be a false negative if the prediction is false
    df["false_neg"] = False
    df["has_impact_neg"] = df["true"] & df["naive_strategy_action"]
    df.loc[df["has_impact_neg"], "false_neg"] = ~df.loc[df["has_impact_neg"], "pred"]

    # False positive with respect to naive strategy action
    # If it is a one-sided PTU and it curtails, it will be a false positive if the prediction is true
    df["false_pos"] = False
    df["has_impact_pos"] = ~df["true"] & df["naive_strategy_action"]
    df.loc[df["has_impact_pos"], "false_pos"] = df.loc[df["has_impact_pos"], "pred"]

    agg_dict = {
        "min_price": "min",         # Min down price
        "max_price": "max",         # Max up price
        "has_impact_neg": "sum",    # Count total possible false negatives
        "has_impact_pos": "sum",    # Count total possible false positives
        "false_neg": "sum",         # Count all the false negatives
        "false_pos": "sum",         # Count all the false positives
    }
    # Make it flat for easy calculations
    flat_df = df.groupby("id").agg(agg_dict)

    false_neg_penalty = flat_df["false_neg"]
    false_pos_penalty = flat_df["false_pos"]
    false_neg_penalty_total = flat_df["has_impact_neg"]
    false_pos_penalty_total = flat_df["has_impact_pos"]

    # Use the example revenue
    if example_revenue:
        energy = 100 / 60  # Example renewable energy
        false_neg_penalty *= flat_df["max_price"] * -energy
        false_pos_penalty *= flat_df["min_price"] * energy
        false_neg_penalty_total *= flat_df["max_price"] * -energy
        false_pos_penalty_total *= flat_df["min_price"] * energy

    false_neg_penalty_sum = false_neg_penalty.sum()
    false_pos_penalty_sum = false_pos_penalty.sum()
    false_neg_penalty_total_sum = false_neg_penalty_total.sum()
    false_pos_penalty_total_sum = false_pos_penalty_total.sum()

    return false_neg_penalty_sum, false_neg_penalty_total_sum, false_pos_penalty_sum, false_pos_penalty_total_sum


def print_penalty_score(results, titles=None):
    """Print the results of the penalty metric."""
    titles = default_titles(results, titles)

    # Create subplots for penalty scores
    make_subplots(
        plot_penalty_score,
        results,
        titles,
        suptitle="Comparison of Penalty Scores",
        figsize=(18, 2)
    )


def show_penalty_score(df: pd.DataFrame, true: pd.Series, pred: pd.Series, ids: pd.Series, example_revenue=False):
    """Show the penalty metric."""
    results = compute_penalty_score(df, true, pred, ids, example_revenue)
    print_penalty_score(results)


def compute_time_diff_score(df: pd.DataFrame, pred: pd.Series, ids: pd.Series):
    """Calculate the time delay between predictions and realtime."""
    df = df.copy()
    df["start_idx"] = True
    df["true"] = df["target_two_sided_ptu_realtime"]
    df["pred"] = pred
    df["id"] = ids
    df["flip"] = detect_flip(df["target_two_sided_ptu_realtime"], df["pred"])
    agg_dict = {
        "start_idx": "idxmax",                      # Get the start index of the PTU
        "flip": "idxmax",                           # The time it flips
        "target_two_sided_ptu_realtime": "idxmax",  # The time it has to flip
        "true": "any",                              # Is the PTU two-sided
        "pred": "any"                               # Prediction PTU two-sided
    }
    flat_df = df.groupby("id").agg(agg_dict)

    # True positives
    true_pos_mask = flat_df["true"] & flat_df["pred"]
    true_pos_time = flat_df.loc[true_pos_mask, "flip"] - flat_df.loc[true_pos_mask, "target_two_sided_ptu_realtime"]
    true_pos_time_avg = true_pos_time.mean()
    true_pos_count = true_pos_mask.sum()

    # Split into positive and negative time differences
    true_pos_time_mask = true_pos_time < 0
    true_pos_time_neg = true_pos_time[true_pos_time_mask].describe()
    true_pos_time_pos = true_pos_time[~true_pos_time_mask].describe()

    # Best case scenario
    true_time = flat_df.loc[flat_df["true"], "start_idx"] - flat_df.loc[flat_df["true"], "target_two_sided_ptu_realtime"]
    true_time_avg = true_time.mean()
    true_count = flat_df["true"].sum()

    time_df = pd.concat([true_pos_time_neg, true_pos_time_pos], axis=1)
    time_df.columns = ["neg", "pos"]

    return time_df, true_pos_time_avg, true_time_avg, true_pos_count, true_count


def print_time_diff_score(results, titles=None):
    """Print the results of the time metric."""
    titles = default_titles(results, titles)

    # Create subplots for time difference analysis
    make_subplots(
        lambda result, title, ax: plot_time_diff_df(result[0], title, ax),
        results,
        titles,
        suptitle="Comparison of Time Difference Scores - Analysis",
        figsize=(18, 3)
    )

    # Create subplots for time difference averages
    make_subplots(
        lambda result, title, ax: plot_time_diff_avg(result[1:], title, ax),
        results,
        titles,
        suptitle="Comparison of Time Difference scores - Average",
        figsize=(18, 2)
    )


def show_time_diff_score(df: pd.DataFrame, pred: pd.Series, ids: pd.Series):
    """Show the time metric."""
    results = compute_time_diff_score(df, pred, ids)
    print_time_diff_score(results)


def compute_metrics(df: pd.DataFrame, true: pd.Series, pred: pd.Series, ids: pd.Series):
    """Show every metric."""
    results = []
    results.append(compute_basic_metrics(true, pred, ids, False))
    results.append(compute_basic_metrics(true, pred, ids))
    results.append(compute_penalty_score(df, true, pred, ids))
    results.append(compute_time_diff_score(df, pred, ids))
    return results


def print_metrics(results):
    """Print every metric."""
    print_basic_metrics(results[0])
    print_basic_metrics(results[1])
    print_penalty_score(results[2])
    print_time_diff_score(results[3])


def show_metrics(df: pd.DataFrame, true: pd.Series, pred: pd.Series, ids: pd.Series):
    """Show every metric."""
    results = compute_metrics(df, true, pred, ids)
    print_metrics(results)


def show_metrics_multi(df: pd.DataFrame, true: pd.Series, preds: list[pd.Series], ids: pd.Series, titles=None):
    """Show every metric for every prediction."""
    results = [compute_metrics(df, true, pred, ids) for pred in preds]
    print_basic_metrics(get_submatrix(results, col_start=0, col_end=1, auto_flat=True), titles)
    print_basic_metrics(get_submatrix(results, col_start=1, col_end=2, auto_flat=True), titles)
    print_penalty_score(get_submatrix(results, col_start=2, col_end=3, auto_flat=True), titles)
    print_time_diff_score(get_submatrix(results, col_start=3, col_end=4, auto_flat=True), titles)


def show_metrics_adjusted(df: pd.DataFrame, true: pd.Series, pred: pd.Series, ids: pd.Series, version="target"):
    """Show every metric for every adjustment."""
    # Adjusted with realtime
    pred_real = adjust_pred_realtime(df[f"{version}_two_sided_ptu_realtime"], pred)

    # Adjusted with consistency
    pred_con = adjust_pred_consistency(pred, ids)

    # Adjusted with consistency and realtime
    pred_con_real = adjust_pred_realtime(df[f"{version}_two_sided_ptu_realtime"], pred_con)

    preds = [pred, pred_real, pred_con, pred_con_real]
    titles = ["Base", "Adjusted Real-time", "Adjusted Consistency", "Adjusted Consistency + Real-time"]
    show_metrics_multi(df, true, preds, ids, titles)
