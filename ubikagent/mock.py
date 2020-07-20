from collections import namedtuple

import numpy as np

BrainParametersMock = namedtuple(
    'BrainParametersMock', [
        'brain_name',
        'vector_action_space_size',
        'vector_action_space_type',
        'vector_observation_space_size',
        'vector_observation_space_type',
    ])

AllBrainInfoMock = namedtuple(
    'AllBrainInfoMock', ['agents', 'local_done', 'rewards', 'vector_observations']
)


class UnityEnvironmentMock:
    """Mock of unityagents.environment.UnityEnvironment for testing.

    Creates also a mock of what the codebase requires
    from UnityEnvironment: BrainParameters, AllBrainInfoMock.

    """
    def __init__(
        self,
        brain_name,
        num_agents,
        vector_observation_space_size,
        vector_observation_space_type,
        vector_action_space_size,
        vector_action_space_type,
        **kwargs
    ):
        """Initializes an instance of UnityEnvironmentMock.

        Args:
            brain_name (str): brain name in the UnityEnvironment
            num_agents (int): number of agents
            vector_observation_space_size (int): state space
            vector_observation_space_type (str): either 'discrete' or 'continuous'
            vector_action_space_size (int): action space size
            vector_action_space_type (str): either 'discrete' or 'continuous'

        """
        self.brain_names = [brain_name]

        self.num_agents = num_agents
        self.vector_action_space_size = vector_action_space_size
        self.vector_action_space_type = vector_action_space_type
        self.vector_observation_space_size = vector_observation_space_size
        self.vector_observation_space_type = vector_observation_space_type

        brain_parameters = BrainParametersMock(
            brain_name,
            vector_action_space_size,
            vector_action_space_type,
            vector_observation_space_size,
            vector_observation_space_type)

        self.brains = {brain_name: brain_parameters}

        if vector_observation_space_type == 'continuous':
            vector_observations = np.random.randn(
                num_agents, vector_observation_space_size)
        elif vector_observation_space_type == 'discrete':
            raise NotImplementedError("discrete state space not implemented")

        agents = list(range(num_agents))
        local_done = [False] * num_agents
        rewards = [0.0] * num_agents

        env_info = AllBrainInfoMock(agents, local_done, rewards, vector_observations)

        self.env_info = {brain_name: env_info}

        if self.vector_action_space_type == 'discrete':
            if self.num_agents > 1:
                expected_shape = (self.space_size,)
            else:
                expected_shape = ()
        else:
            expected_shape = (self.num_agents, self.vector_action_space_size)
        self.expected_action_shape = expected_shape

    def reset(self, **kwargs):
        """Provides `reset()`, but only returns static AllBrainInfo object.

        Returns:
            AllBrainInfo object

        """
        return self.env_info

    def step(self, action, **kwargs):
        """Provides `step()`, but only validates input shape.

        Args:
            action (int, or np.ndarray): action from the agent

        Raises:
            TypeError if the shape of action does not match expectation.

        Returns:
            AllBrainInfo object

        """
        if np.shape(action) == self.expected_action_shape:
            return self.env_info

        raise TypeError("action size and expectation don't match")

    def close(self):
        """Provides `close()`, and does and returns nothing."""
        return

    def seed(self, seed):
        """Provides `seed()`, but does nothing."""
        return
