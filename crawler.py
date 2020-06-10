#!/usr/bin/env python

import os
import argparse
from unityagents import UnityEnvironment

from ubikagent.interaction import UnityInteraction
from ubikagent.helper import (
    get_model_dir, create_model_dir, save_graph, save_history)
from ubikagent.agent import DDPGAgent
from ubikagent.agent import RandomAgent

ENV_PATH = './environments/Crawler_Linux/Crawler.x86_64'

MODEL_PARAMS = {
    'lr_actor': 3e-3,
    'layers_actor': [128, 64],
    'lr_critic': 3e-3,
    'layers_critic': [64, 32, 32],
    'batch_size': 128,
    'tau': 2e-2,
    'gamma': 0.99,
    'replay_buffer_size': 1e5,
    'seed': 42,
}

TRAINING_PARAMS = {
    'num_episodes': 300,
    'max_time_steps': 1000,
    'score_target': 2000.0,
}


def train(modelname):

    # create environment
    env = UnityEnvironment(file_name=ENV_PATH, no_graphics=True)

    # create an agent
    state_size, action_size, num_agents = UnityInteraction.stats(env)
    agent = DDPGAgent(state_size, action_size, num_agents, **MODEL_PARAMS)

    # and create an interaction between them
    sim = UnityInteraction(agent, env)

    if modelname is not None:
        create_model_dir(modelname)

    history = sim.run(**TRAINING_PARAMS)

    if modelname is not None:
        modelfile = os.path.join(get_model_dir(modelname))
        agent.save(modelfile)
        save_history(modelname, history)
        save_graph(modelname, history['score'])

    env.close()

def run(modelname):

    # create environment
    env = UnityEnvironment(file_name=ENV_PATH, no_graphics=False)

    # create an agent
    state_size, action_size, num_agents = UnityInteraction.stats(env)
    agent = DDPGAgent(state_size, action_size, num_agents, **MODEL_PARAMS)
    modelfile = os.path.join(get_model_dir(modelname))
    agent.load(modelfile)

    # create train or run loop
    sim = UnityInteraction(agent, env)
    sim.run(learn=False)
    env.close()

def random_run():

    # create environment
    env = UnityEnvironment(file_name=ENV_PATH, no_graphics=False)

    # create an agent
    state_size, action_size, num_agents = UnityInteraction.stats(env)
    agent = RandomAgent(
        state_size, action_size, action_type='continuous', num_agents=num_agents)

    # create train or run loop
    sim = UnityInteraction(agent, env)
    sim.run(learn=False)
    env.close()

def main():

    parser = argparse.ArgumentParser(
        description='Runs or trains an agent in Unity Crawler environment.')

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
