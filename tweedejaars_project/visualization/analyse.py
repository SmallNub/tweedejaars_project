import matplotlib.pyplot as plt
import pandas as pd

from tweedejaars_project.visualization.visualize import add_ids_arg, add_show_arg


@add_show_arg(True)
@add_ids_arg()
def plot_against_index(feature: pd.Series):
    """Plot a feature against the index"""
    plt.plot(feature, label=feature.name)


@add_show_arg(True)
def plot_scatter(feature_x: pd.Series, feature_y: pd.Series):
    """Plot two features in a scatter plot"""
    plt.scatter(feature_x, feature_y)
    plt.xlabel(feature_x.name)
    plt.ylabel(feature_y.name)
