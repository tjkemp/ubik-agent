#!/usr/bin/env python

import os
import argparse
from unityagents import UnityEnvironment
import numpy as np

from agent.interaction import UnityInteraction
from agent.helper import get_model_dir, create_model_dir, save_scores
from agent.ddpg import DDPGAgent
from agent.agent import RandomAgent

ENV_PATH = './environments/Crawler_Linux/Crawler.x86_64'
SAVEFILE = 'checkpoint.pth'

MODEL_PARAMS = {
    'lr_actor': 3e-3,
    'lr_critic': 3e-3,
    'batch_size': 512,
    'tau': 2e-1,
    'gamma': 0.99,
    'replay_buffer_size': 1e5,
    'seed': 42,
}

TRAINING_PARAMS = {
    'num_episodes': 100,
    'max_time_steps': 1000,
    'target_score': 400.0,
}


def main(mode, modelname):

    # create environment
    no_graphics = True if mode == 'train' else False
    env = UnityEnvironment(file_name=ENV_PATH, no_graphics=no_graphics)

    # create an agent
    state_size, action_size, num_agents = UnityInteraction.stats(env)

    if mode != 'random':
        agent = DDPGAgent(state_size, action_size, num_agents, **MODEL_PARAMS)
    else:
        agent = RandomAgent(
            state_size, action_size, action_type='continuous', num_agents=num_agents)

    # create train or run loop
    sim = UnityInteraction(agent, env)

    if mode == 'train':
        create_model_dir(modelname)
        scores = sim.train(**TRAINING_PARAMS)
        modelfile = os.path.join(get_model_dir(modelname), SAVEFILE)
        agent.save(modelfile)
        save_scores(modelname, scores)
    elif mode == 'random':
        sim.run()
    elif mode == 'run':
        modelfile = os.path.join(get_model_dir(modelname), SAVEFILE)
        agent.load(modelfile)
        sim.run()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Runs or trains an agent in Unity Reacher environment.')
    parser.add_argument(
        'mode', choices=['train', 'run', 'random'], help="either 'train' or 'run' or 'random'")
    parser.add_argument(
        'modelname', help="name of the agent's model in models directory")
    args = parser.parse_args()
    main(args.mode, args.modelname)
