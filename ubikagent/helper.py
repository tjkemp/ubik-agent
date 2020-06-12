import os
import json
import argparse
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


def print_episode_statistics(history, multi_agent=False):
    """Prints a single row of statistics on episode performance."""

    human_readable = {
        'episode_length': "Steps",
    }

    print(f"Episode {history.num_episodes}", end='')

    for key in history.keys():

        metric_value = history.get_latest(key)
        if key in human_readable:
            metric_name = human_readable[key]
        else:
            metric_name = key.capitalize()

        print(f" \t{metric_name}: {metric_value:.2f}", end='')

    print()


def print_target_reached(history):
    """Prints a notification that target score has been reached."""
    print(f"\nTarget score reached in {history.num_episodes:d} episodes!")


def parse_and_run(project):
    """Parses cli arguments and runs a corresponding class and method.

    This function is used in main() to provide command line interface
    to agent classes, so that, for example `python -m examples.banana train`
    could be evoked to call `train()` in which ever solution class is
    defined in the `main()` function of *examples.banana* module.

    In the future this may be refactored into class which dynamically
    creates arguments.

    Args:
        project (object): an instantiated class in which the project is defined

    Side effects:
        Runs a method in a whichever class is given in the cli arguments.

    """
    parser = argparse.ArgumentParser(
        description=f"Runs or trains an agent in an OpenGym environment.")

    subparsers = parser.add_subparsers(
        title='method', dest='method', help='additional help')

    parser_train = subparsers.add_parser(
        'train', help='train an agent')
    parser_train.add_argument(
        'modelname',
        nargs='?',
        help="directory name in models where to save the agent model")

    parser_run = subparsers.add_parser(
        'run', help='run environment with trained agent')
    parser_run.add_argument(
        'modelname',
        help="directory name in models from where to load the agent model")

    parser_random = subparsers.add_parser(
        'random', help='run with randomly acting agent')
    parser_random.set_defaults(modelname=None)

    parser_optimize = subparsers.add_parser(
        'optimize', help='run hyperparameter optimization')
    parser_optimize.set_defaults(modelname=None)

    parser_interactive = subparsers.add_parser(
        'interactive', help='interact with the environment')
    parser_interactive.set_defaults(modelname=None)

    args = parser.parse_args()

    if args.method is None:
        parser.print_help()
    else:
        method_name = args.method
        method_args = vars(args)
        del method_args['method']
        try:
            method = getattr(project, method_name)
            method(**method_args)
        except AttributeError as err:
            print(f"Error while calling the method '{method}': {err}")
