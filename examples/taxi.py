#!/usr/bin/env python

import os

import gym
import optuna

from ubikagent import Project, Interaction
from ubikagent.agent import SarsaAgent
from ubikagent.callback import TargetScore


class Taxi(Project):

    ENV_ID = 'Taxi-v3'
    AGENT_CLASS = SarsaAgent
    SEED = 1234
    MODEL_PARAMS = {
        'alpha': 0.5,
        'epsilon': 0.92,
        'epsilon_decay': 0.91,
        'epsilon_min': 0.02,
        'gamma': 0.95,
        'algorithm': 'q-learning',
    }
    TRAINING_PARAMS = {
        'num_episodes': 500,
        'max_time_steps': 200,
    }

    def optimize(self, modelname):

        # create environment
        env = gym.make(self.ENV_ID)

        def objective(trial):

            num_episodes = 300
            max_time_steps = 200

            alpha = trial.suggest_uniform('alpha', 0.01, 0.5)
            epsilon = trial.suggest_uniform('epsilon', 0.9, 1.0)
            epsilon_decay = trial.suggest_uniform('epsilon_decay', 0.8, 1.0)
            epsilon_min = trial.suggest_uniform('epsilon_min', 0.01, 0.1)
            gamma = trial.suggest_uniform('gamma', 0.5, 1.0)

            # create an agent
            agent = SarsaAgent(
                env.observation_space,
                env.action_space,
                self.SEED,
                alpha=alpha,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
                gamma=gamma)

            # and train the agent
            sim = Interaction(agent, env, self.SEED)
            calculate_score = TargetScore(100, 0.0)
            history = sim.run(
                num_episodes=num_episodes,
                max_time_steps=max_time_steps,
                verbose=0,
                callbacks=[calculate_score])

            return history['score'][-1]

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        trial = study.best_trial
        print('Best score: {}'.format(trial.value))
        print("Best hyperparameters: {}".format(trial.params))

if __name__ == "__main__":
    project = Taxi().cli()
