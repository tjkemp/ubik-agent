#!/usr/bin/env python

import os
import argparse

import gym
import optuna

from ubikagent import Interaction
from ubikagent.helper import (
    get_model_dir, create_model_dir, save_graph, save_history, parse_and_run)
from ubikagent.agent import RandomAgent
from ubikagent.agent import SarsaAgent

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

class Taxi:

    def optimize(self, modelname):

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
            state_size = env.observation_space
            action_size = env.action_space
            agent = SarsaAgent(
                action_size.n,
                alpha=alpha,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
                gamma=gamma)

            # and train the agent
            sim = Interaction(agent, env)
            history = sim.run(
                num_episodes=num_episodes,
                max_time_steps=max_time_steps,
                verbose=0)

            return history['score'][-1]

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        trial = study.best_trial
        print('Best score: {}'.format(trial.value))
        print("Best hyperparameters: {}".format(trial.params))

    def train(self, modelname):

        # create environment
        env = gym.make(ENV_ID)

        # create an agent
        state_size = env.observation_space
        action_size = env.action_space
        agent = SarsaAgent(action_size.n, **MODEL_PARAMS)

        # and create an interaction between them
        sim = Interaction(agent, env)

        if modelname is not None:
            create_model_dir(modelname)

        history = sim.run(**TRAINING_PARAMS)

        if modelname is not None:
            modeldir = os.path.join(get_model_dir(modelname))
            agent.save(modeldir)
            save_history(modelname, history)
            save_graph(modelname, history['score'])

        env.close()

    def run(self, modelname):

        # create environment
        env = gym.make(ENV_ID)

        # create an agent
        state_size = env.observation_space
        action_size = env.action_space
        agent = SarsaAgent(
            action_size.n,
            epsilon=0.1,
            algorithm=MODEL_PARAMS['algorithm'])
        modelfile = os.path.join(get_model_dir(modelname))
        agent.load(modelfile)

        # run simulation
        sim = Interaction(agent, env)
        sim.run()

        env.close()

    def random(self, modelname):

        # create environment
        env = gym.make(ENV_ID)

        # create an agent
        state_size = env.observation_space
        action_size = env.action_space
        agent = RandomAgent(state_size, action_size)

        # create train or run loop
        sim = Interaction(agent, env)
        sim.run()
        env.close()

if __name__ == "__main__":
    project = Taxi()
    args = parse_and_run(project)
