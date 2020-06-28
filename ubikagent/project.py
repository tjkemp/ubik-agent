import os

import gym
from gym.spaces import flatdim

from ubikagent import Interaction
from ubikagent.agent import RandomAgent
from ubikagent.helper import (
    get_model_dir, create_model_dir, save_graph, save_history, parse_and_run)
from ubikagent.callback import TargetScore


gym.logger.set_level(40)


class BaseProject:
    pass

class Project(BaseProject):

    ENV_ID = None
    AGENT_CLASS = None
    MODEL_PARAMS = {}
    TRAINING_PARAMS = {}

    def __init__(self):

        if self.ENV_ID is None:
            raise Exception("define ENV_ID so the Gym environment can be instantiated")
        if self.AGENT_CLASS is None:
            raise Exception("define ENV_ID so the Gym environment can be instantiated")

    def train(self, modelname):

        # create environment
        env = gym.make(self.ENV_ID)

        # create an agent
        state_size = flatdim(env.observation_space)  # noqa: F841
        action_size = flatdim(env.action_space)
        print("dims ", state_size, action_size)
        agent = self.AGENT_CLASS(state_size, action_size, **self.MODEL_PARAMS)

        # and create an interaction between them
        sim = Interaction(agent, env)

        calculate_score = TargetScore(100, 0.0)
        history = sim.run(**self.TRAINING_PARAMS, callbacks=[calculate_score])

        # _save and close the environment
        self._save(modelname, agent, history)
        env.close()

    def run(self, modelname):

        # create environment
        env = gym.make(self.ENV_ID)

        # create an agent
        state_size = env.observation_space  # noqa: F841
        action_size = env.action_space

        agent = self.AGENT_CLASS(
            state_size,
            action_size,
            **self.MODEL_PARAMS)

        self._load(modelname, agent)

        # run simulation
        sim = Interaction(agent, env)
        sim.run()

        env.close()

    def random(self, modelname):

        # create environment
        env = gym.make(self.ENV_ID)

        # create an agent
        agent = RandomAgent(
            env.observation_space, env.action_space)

        # create an interaction
        sim = Interaction(agent, env)
        sim.run()
        env.close()

    def cli(self):
        args = parse_and_run(self)

    def _load(self, modelname, agent):

        modelfile = os.path.join(get_model_dir(modelname))
        agent._load(modelfile)

    def _save(self, modelname, agent, history):

        if modelname is not None:
            create_model_dir(modelname)
            modeldir = os.path.join(get_model_dir(modelname))
            agent._save(modeldir)
            save_history(modelname, history)
            save_graph(modelname, history['score'])
