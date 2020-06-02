#!/usr/bin/env python

import os
import argparse
from unityagents import UnityEnvironment
import numpy as np

from agent.interaction import UnityInteraction
from agent.helper import (
    get_model_dir, create_model_dir, save_graph, save_history)
from agent import DQNAgent
from agent.agent import RandomAgent

ENV_PATH = './environments/Banana_Linux/Banana.x86_64'

MODEL_PARAMS = {
    'learning_rate': 5e-4,
    'batch_size': 64,
    'tau': 1e-3,
    'gamma': 0.99,
    'update_interval': 4,
    'replay_buffer_size': 1e5,
    'seed': 4,
}

TRAINING_PARAMS = {
    'num_episodes': 500,
    'max_time_steps': 300,
    'score_target': 13.0,
}

def train(modelname):

    # create environment
    env = UnityEnvironment(file_name=ENV_PATH, no_graphics=True)

    # create an agent
    state_size, action_size, num_agents = UnityInteraction.stats(env)
    agent = DQNAgent(state_size, action_size, num_agents, **MODEL_PARAMS)

    # and create an interaction between them
    sim = UnityInteraction(agent, env)

    if modelname is not None:
        create_model_dir(modelname)

    history = sim.train(**TRAINING_PARAMS)

    if modelname is not None:
        modeldir = os.path.join(get_model_dir(modelname))
        agent.save(modeldir)
        save_history(modelname, history)
        save_graph(modelname, history['score'])

    env.close()

def run(modelname):

    # create environment
    env = UnityEnvironment(file_name=ENV_PATH, no_graphics=False)

    # create an agent
    state_size, action_size, num_agents = UnityInteraction.stats(env)
    agent = DQNAgent(state_size, action_size, num_agents, **MODEL_PARAMS)
    modelfile = os.path.join(get_model_dir(modelname))
    agent.load(modelfile)

    # run simulation
    sim = UnityInteraction(agent, env)
    sim.run()

    env.close()

def random_run():

    # create environment
    env = UnityEnvironment(file_name=ENV_PATH, no_graphics=False)

    # create an agent
    state_size, action_size, num_agents = UnityInteraction.stats(env)
    agent = RandomAgent(
        state_size, action_size, action_type='discrete', num_agents=num_agents)

    # create train or run loop
    sim = UnityInteraction(agent, env)
    sim.run()
    env.close()

def main():

    parser = argparse.ArgumentParser(
        description='Runs or trains an agent in Unity BananaCollector environment.')

    subparsers = parser.add_subparsers(
        title='subcommands', dest='subcommand', help='additional help   ')

    parser_train = subparsers.add_parser('train', help='train an agent')
    parser_train.add_argument(
        'modelname', nargs='?', help="directory name in models where to save the agent model")

    parser_run = subparsers.add_parser(
        'run', help='run environment with trained agent')
    parser_run.add_argument(
        'modelname', help="directory name in models from where to load the agent model")

    parser_random = subparsers.add_parser(
        'random', help='run environment with randomly acting agent')
    parser_random.set_defaults(modelname=None)

    args = parser.parse_args()

    if args.subcommand == 'train':
        train(args.modelname)
    elif args.subcommand == 'run':
        run(args.modelname)
    elif args.subcommand == 'random':
        random_run()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
