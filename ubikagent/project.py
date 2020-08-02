import os
import inspect
import argparse

import gym
from gym.spaces import flatdim

from ubikagent import Interaction
from ubikagent.agent import RandomAgent
from ubikagent.helper import (
    get_model_dir, create_model_dir, save_graph, save_parameters, save_history)
from ubikagent.callback import TargetScore
from ubikagent.introspection import get_methods

gym.logger.set_level(40)


class BaseProject:
    pass

class Project(BaseProject):

    ENV_ID = None
    AGENT_CLASS = None
    MODEL_PARAMS = {}
    TRAINING_PARAMS = {}
    SEED = 1234

    def __init__(self):

        super().__init__()
        if self.ENV_ID is None:
            raise Exception("define ENV_ID (str) so the Gym environment can be instantiated")
        if self.AGENT_CLASS is None:
            raise Exception("define AGENT_CLASS (class) so the agent can be instantiated")

    def train(self, modelname=None):

        # create environment
        env = gym.make(self.ENV_ID)

        # create an agent
        agent = self.AGENT_CLASS(
            env.observation_space,
            env.action_space,
            self.SEED,
            **self.MODEL_PARAMS)

        # and create an interaction between them
        sim = Interaction(agent, env, self.SEED)

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
            self.SEED,
            **self.MODEL_PARAMS)

        self._load(modelname, agent)

        # run simulation
        sim = Interaction(agent, env, self.SEED)
        sim.run()

        env.close()

    def random(self):

        # create environment
        env = gym.make(self.ENV_ID)

        # create an agent
        agent = RandomAgent(
            env.observation_space, env.action_space, self.SEED)

        # create an interaction
        sim = Interaction(agent, env, self.SEED)
        sim.run()
        env.close()

    def cli(self):
        """Creates command-line argument parser from class definition, parses
        the arguments, and finally runs a corresponding method in the class.

        This function is used in a custom `main()` to provide a command line
        interface to a class, so that, for example, `python -m examples.banana
        train` could be evoked to call `train()` on `Project` subclass
        `Banana` defined in the `main()` of module *examples.banana*.

        See `examples` package for usage examples.

        Side effects:
            Runs a method in a whichever class is given in the cli arguments.

        """
        parser = argparse.ArgumentParser(
            description="Calls a method in subclass of Project, for example, to train an agent in an Gym environment.")

        subparsers = parser.add_subparsers(
            title='method', dest='method', help='a method in the class')

        methods_and_args = get_methods(self)

        for method, arguments in methods_and_args.items():

            subparser = subparsers.add_parser(
                method, help=method)

            for argument in arguments:
                param_name, is_kwarg, param_default, param_type, param_doc = argument

                msg_default_value = f"default is {param_default}"
                if param_type is bool:
                    bool_parser = subparser.add_mutually_exclusive_group(required=False)
                    bool_parser.add_argument(
                        '--' + param_name,
                        dest=param_name,
                        action='store_true',
                        help=f"set '{param_name}' as True, " + msg_default_value)
                    bool_parser.add_argument(
                        '--no-' + param_name,
                        dest=param_name,
                        action='store_false',
                        help=f"set '{param_name}' as False, " + msg_default_value)
                    parser.set_defaults(param_name=param_default)
                else:
                    name = '--' + param_name if is_kwarg else param_name
                    if param_doc is None and is_kwarg is False:
                        msg_help = "mandatory argument"
                    elif param_doc is None and is_kwarg is True:
                        msg_help = msg_default_value
                    else:
                        msg_help = param_doc + ", " + msg_default_value
                    subparser.add_argument(
                        name,
                        type=param_type,
                        default=param_default,
                        help=msg_help)

        args = parser.parse_args()

        if args.method is None:
            parser.print_help()
        else:
            method_name = args.method
            method_args = vars(args)
            del method_args['method']
            try:
                method = getattr(self, method_name)
                method(**method_args)
            except AttributeError as err:
                print(f"Error while calling the method '{method}': {err}")

    def _load(self, modelname, agent):

        modelfile = os.path.join(get_model_dir(modelname))
        agent.load(modelfile)

    def _save(self, modelname, agent, history):

        if modelname is not None:
            create_model_dir(modelname)
            modeldir = os.path.join(get_model_dir(modelname))
            agent.save(modeldir)
            save_parameters(
                modeldir,
                {
                    'MODEL_PARAMS': self.MODEL_PARAMS,
                    'TRAINING_PARAMS': self.TRAINING_PARAMS
                })
            save_history(modelname, history)
            save_graph(modelname, history['score'])
