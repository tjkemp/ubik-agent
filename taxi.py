#!/usr/bin/env python

import os
import argparse

import gym
import optuna

from agent.interaction import GymInteraction
from agent.helper import (
    get_model_dir, create_model_dir, save_graph, save_history)
from agent.agent import RandomGymAgent
from agent.sarsa import SarsaAgent

MODEL_PARAMS = {
    'alpha': 0.5,
    'epsilon': 0.92,
    'epsilon_decay': 0.91,
    'epsilon_min': 0.02,
    'gamma': 0.95,
    'algorithm': 'sarsamax',
}

TRAINING_PARAMS = {
    'num_episodes': 500,
    'max_time_steps': 200
}

ENV_ID = 'Taxi-v3'

def optimize(modelname):

    # create environment
    env = gym.make(ENV_ID)

    def objective(trial):

        num_episodes = 300
        max_time_steps = 200

        alpha = trial.suggest_uniform('alpha', 0.01, 0.5)
        epsilon = trial.suggest_uniform('epsilon', 0.9, 1.0)
        epsilon_decay = trial.suggest_uniform('epsilon_decay', 0.8, 1.0)
        epsilon_min = trial.suggest_uniform('epsilon_min', 0.01, 0.1)
        gamma = trial.suggest_uniform('gamma', 0.5, 1.0)

        # create an agent
        state_size, action_size, num_agents = GymInteraction.stats(env)
        agent = SarsaAgent(
            action_size.n,
            alpha=alpha,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            gamma=gamma)

        # and train the agent
        sim = GymInteraction(agent, env)
        history = sim.train(
            num_episodes=num_episodes,
            max_time_steps=max_time_steps,
            verbose=0)

        return history['score'][-1]

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    trial = study.best_trial
    print('Best score: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))


def train(modelname):

    # create environment
    env = gym.make(ENV_ID)

    # create an agent
    state_size, action_size, num_agents = GymInteraction.stats(env)
    agent = SarsaAgent(action_size.n, **MODEL_PARAMS)

    # and create an interaction between them
    sim = GymInteraction(agent, env)

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
    env = gym.make(ENV_ID)

    # create an agent
    state_size, action_size, num_agents = GymInteraction.stats(env)
    agent = SarsaAgent(
        action_size.n,
        epsilon=0.1,
        algorithm=MODEL_PARAMS['algorithm'])
    modelfile = os.path.join(get_model_dir(modelname))
    agent.load(modelfile)

    # run simulation
    sim = GymInteraction(agent, env)
    sim.run()

    env.close()

def random_run():

    # create environment
    env = gym.make(ENV_ID)

    # create an agent
    state_size, action_size, num_agents = GymInteraction.stats(env)
    agent = RandomGymAgent(
        state_size, action_size, action_type='discrete', num_agents=num_agents)

    # create train or run loop
    sim = GymInteraction(agent, env)
    sim.run()
    env.close()

def main():

    parser = argparse.ArgumentParser(
        description='Runs or trains an agent in OpenAI Gym environment.')

    subparsers = parser.add_subparsers(
        title='subcommands', dest='subcommand', help='additional help')

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

    parser_train = subparsers.add_parser('optimize', help='search for best hyperparameters')
    parser_train.add_argument(
        'modelname', nargs='?', help="directory name in models where to save the agent model")

    args = parser.parse_args()

    if args.subcommand == 'train':
        train(args.modelname)
    elif args.subcommand == 'run':
        run(args.modelname)
    elif args.subcommand == 'random':
        random_run()
    elif args.subcommand == 'optimize':
        optimize(args.modelname)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
