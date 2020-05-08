#!/usr/bin/env python

import os
import argparse
from unityagents import UnityEnvironment
import numpy as np

from agent.interaction import UnityInteraction
from agent.helper import get_model_dir, create_model_dir, save_scores
from agent import DQNAgent

ENV_PATH = './environments/Banana_Linux/Banana.x86_64'
SAVEFILE = 'checkpoint.pth'

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
    'num_episodes': 800,
    'max_time_steps': 300,
    'target_score': 13.0,
}


def main(mode, modelname):

    # create environment
    no_graphics = True if mode == 'train' else False
    env = UnityEnvironment(file_name=ENV_PATH, no_graphics=no_graphics)

    # create an agent
    state_size, action_size, num_agents = UnityInteraction.stats(env)
    agent = DQNAgent(state_size, action_size, num_agents, **MODEL_PARAMS)

    # create train or run loop
    sim = UnityInteraction(agent, env)

    if mode == 'train':
        create_model_dir(modelname)
        scores = sim.train(**TRAINING_PARAMS)
        modelfile = os.path.join(get_model_dir(modelname), SAVEFILE)
        agent.save(modelfile)
        save_scores(modelname, scores)

    elif mode == 'run':
        modelfile = os.path.join(get_model_dir(modelname), SAVEFILE)
        agent.load(modelfile)
        sim.run()

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Runs or trains an agent in Unity BananaCollector environment.')
    parser.add_argument(
        'mode', choices=['train', 'run'], help="either 'train' or 'run'")
    parser.add_argument(
        'modelname', help="name of the agent's model in models directory")
    args = parser.parse_args()
    main(args.mode, args.modelname)
