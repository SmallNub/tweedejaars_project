import matplotlib.pyplot as plt
import pandas as pd


ORDER = ['true', 'pred']


def add_show_arg(show_default=False):
    """Decorator to add a plot show arg"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            show = kwargs.pop('show', show_default)

            result = func(*args, **kwargs)

            if show:
                show_graph()

            return result
        return wrapper
    return decorator


def add_ids_arg(ids_default=None):
    """Decorator to add an arg to use job ids"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            ids = kwargs.pop('ids', ids_default)

            result = func(*args, **kwargs)

            if ids is not None:
                add_id_lines(ids)

            return result
        return wrapper
    return decorator


def show_graph():
    """Show the graphs and reorder graphs"""
    plt.gcf().set_size_inches(20, 8)
    change_order()
    plt.show()


def change_order(order=ORDER):
    """Reorder the graphs"""
    handles, labels = plt.gca().get_legend_handles_labels()

    front = 0
    for label in order:
        try:
            index = labels.index(label)
        except ValueError:
            continue

        labels.insert(front, labels.pop(index))
        handles.insert(front, handles.pop(index))
        front += 1

    plt.legend(handles, labels)


def increment_color_cycle():
    """Increment the default color cycle of matplotlib"""
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    new_colors = colors[1:] + colors[:1]
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', new_colors)


def add_id_lines(ids: pd.Series):
    """Add the job id lines"""
    for index in ids.drop_duplicates(keep='first').index:
        plt.axvline(x=index, color='r', linestyle='--')
        plt.text(index, plt.ylim()[1], f'Job ID: {ids[index]}', rotation=90, verticalalignment='bottom')
