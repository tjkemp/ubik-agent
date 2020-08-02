import os
import json
import argparse

import matplotlib.pyplot as plt

from ubikagent import exception


def get_model_dir(model_name, model_dir='models'):
    """Convenience function to get the directory where to save the model."""
    return os.path.join(os.getcwd(), model_dir, model_name)


def create_model_dir(model_name, model_dir='models'):
    """Convenience function to create a directory where to save the model."""
    path = os.path.join(os.getcwd(), model_dir, model_name)
    try:
        os.makedirs(path, exist_ok=False)
    except FileExistsError:
        raise exception.UbikFileExistsError(
            f"models directory {path} exists already")

def save_graph(modelname, graph_data, ylabel='scores', filename='scores.png'):
    """Convenience function to save graphs (e.g. scores).

    Args:
        graph_data (list): a list of numbers to plot

    """
    plt.plot(list(range(1, len(graph_data) + 1)), graph_data)
    plt.ylabel(ylabel)
    plt.xlabel('episode')
    save_path = os.path.join(get_model_dir(modelname), filename)
    plt.savefig(save_path)


def save_history(model_name, history, filename='history.json'):
    """Save json output of class `History` into a file."""

    str_data = json.dumps(history, ensure_ascii=False, indent=4)
    save_path = os.path.join(get_model_dir(model_name), filename)
    with open(save_path, 'w+') as output_file:
        output_file.write(str_data)


def save_parameters(model_name, params, filename='parameters.json'):
    """Save training parameters into a file."""

    str_data = json.dumps(params, ensure_ascii=False, indent=4)
    save_path = os.path.join(get_model_dir(model_name), filename)
    with open(save_path, 'w+') as output_file:
        output_file.write(str_data)


def print_episode_statistics(history, multi_agent=False):
    """Prints a single row of statistics on episode performance."""

    human_readable = {
        'episode_length': "steps",
    }

    print(f"episode {history.num_episodes}", end='')

    for key in history.keys():

        metric_value = history.get_latest(key)
        if key in human_readable:
            metric_name = human_readable[key]
        else:
            metric_name = key

        print(f" \t{metric_name}: {metric_value:.3g}", end='')

    print()


def print_target_reached(history):
    """Prints a notification that target score has been reached."""
    print(f"\nTarget score reached in {history.num_episodes:d} episodes!")
