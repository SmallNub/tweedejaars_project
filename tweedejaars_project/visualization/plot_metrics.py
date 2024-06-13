import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def default_titles(results, titles):
    """Generate default titles if needed."""
    if titles is None:
        titles = list(range(len(results)))
    return titles


def make_subplots(subplot_func, results, titles, suptitle, figsize):
    """Make subplots using results."""
    # Check if there are multiple results
    if isinstance(results[0], (tuple, list)):
        single = False
    else:
        single = True
        results = [results]

    # Create subplots
    fig, axes = plt.subplots(1, len(results), figsize=figsize)

    if single:
        axes = [axes]

    # Plot each subplot
    for result, title, ax in zip(results, titles, axes):
        subplot_func(result, title, ax)

    show_subplots(fig, suptitle)


def show_subplots(fig, title):
    """Show the subplots."""
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_df(df, title, ax, text=None):
    """Custom table plot for a small df."""
    ax.axis("tight")
    ax.axis("off")
    df = df.map(lambda x: f"{np.round(x, 2):.2f}")
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.3)
    ax.set_title(title)

    if text is not None:
        # Add additional text below the table
        ax.text(0.5, 0.2, text, ha='center', va='top', transform=ax.transAxes, fontsize=12)


def plot_classification_report(report, title, ax):
    """Custom plot for a classification report."""
    classes = list(report.keys())
    metrics = ["precision", "recall", "f1-score", "support"]

    data = []
    for cls in classes:
        row = []
        if isinstance(report[cls], dict):
            for metric in metrics:
                row.append(report[cls].get(metric, ""))
        else:  # for "accuracy"
            row = [report[cls]] * 4
        data.append(row)

    df = pd.DataFrame(data, columns=metrics, index=classes)
    plot_df(df, title, ax)


def plot_confusion_matrix(cm, title, ax):
    """Plot a confusion matrix."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["False", "True"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)


def plot_penalty_score(score, title, ax):
    """Plot penalty score."""
    index_names = ["False neg", "False pos"]
    column_names = ["Percentage", "Prediction", "Maximum"]
    data = [[score[0] / score[1], score[0], score[1]],
            [score[2] / score[3], score[2], score[3]]]
    df = pd.DataFrame(data, columns=column_names, index=index_names)
    plot_df(df, title, ax)


def plot_income_score(score, title, ax):
    """Plot income score."""
    index_names = ["Prediction", "Maximum"]
    column_names = ["Naive income", "Model income", "Added value"]
    data = [[score[0], score[1], score[2]],
            [score[0], score[3], score[4]]]
    df = pd.DataFrame(data, columns=column_names, index=index_names)
    plot_df(df, title, ax, f"Added value percentage to max: {score[2] / score[4]:.3f}")


def plot_time_diff_df(score, title, ax):
    """Plot time difference score."""
    index_names = score.index
    column_names = score.columns
    data = score
    df = pd.DataFrame(data, columns=column_names, index=index_names)
    plot_df(df, title, ax)


def plot_time_diff_avg(score, title, ax):
    """Plot time difference score."""
    index_names = ["Time delay", "True pos count"]
    column_names = ["Pred", "Max"]
    data = [[score[0], score[1]],
            [score[2], score[3]]]
    df = pd.DataFrame(data, columns=column_names, index=index_names)
    plot_df(df, title, ax)
