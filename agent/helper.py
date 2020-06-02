import os
import json
import matplotlib.pyplot as plt

def get_model_dir(model_name, model_dir='models'):
    """Convenience function to get the directory where to save the model."""
    return os.path.join(os.getcwd(), model_dir, model_name)

def create_model_dir(model_name, model_dir='models'):
    """Convenience function to create a directory where to save the model."""
    os.mkdir(os.path.join(os.getcwd(), model_dir, model_name))

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
    """Save json output of class `History` in into a file."""

    str_data = json.dumps(history, ensure_ascii=False, indent=4)
    save_path = os.path.join(get_model_dir(model_name), filename)
    with open(save_path, 'w+') as output_file:
        output_file.write(str_data)
