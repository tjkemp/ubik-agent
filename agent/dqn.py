import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .agent import Agent
from .buffer import ReplayBuffer
from .model import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent(Agent):
    """Deep Q-Learning algorithm for discrete actions spaces.

    Deep Q-Learning algorithm first introduced by DeepMind's
    [research paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

    Q-Learning is a model-free reinforcement learning algorithm to learn a policy
    for an agent. In Deep Q-Learning, a neural network represents the Q. More
    specifically, the DQN takes the state as an input and returns the corresponding
    predicted action values for each possible actions.

    """
    def __init__(
            self,
            state_size,
            action_size,
            num_agents,
            learning_rate=5e-4,
            batch_size=64,
            tau=1e-3,
            gamma=0.99,
            update_interval=4,
            replay_buffer_size=1e5,
            seed=42,
            eps_start=1.0,
            eps_end=0.01,
            eps_decay=0.995,
    ):
        """Initializes an Agent object.

        Args:
            state_size (int): required, dimension of each state
            action_size (int): required, dimension of each discrete action
            num_agents (int): required, number of agents in the simulation
            learning_rate (int): learning rate for the neural network
            batch_size (int): batch size for training the neural network
            tau (int): soft update of target parameters
            gamma (float): dicount factor, between 0.0 and 1.0
            update_interval (int): how often to update the network
            replay_buffer_size (int): length of learning history from which to learn
            seed (int): random seed
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon

        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.update_interval = update_interval
        self.replay_buffer_size = int(replay_buffer_size)
        self.seed = random.seed(seed)

        self.layer_sizes = (state_size, round(state_size / 2))
        self.qnetwork_local = QNetwork(
            state_size, action_size, layer_sizes=self.layer_sizes, seed=seed
        ).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, layer_sizes=self.layer_sizes, seed=seed
        ).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        self.memory = ReplayBuffer(self.replay_buffer_size, batch_size, seed)

        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.timestep = 0

    def new_episode(self):
        return

    def act(self, state, eps=None):
        """Returns action for given state as per current policy.

        Uses epsilon-greedy action selection. When epsilon is 1.0,
        the action is totally random. When epsilon is 0.0 the returned
        action is always the best action according the current policy.

        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection,
                if set, overrides the internal epsilon

        Returns:
            int: chosen action
        """

        epsilon = eps if eps is not None else self.epsilon

        if random.random() > epsilon:

            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            return np.argmax(action_values.cpu().data.numpy())

        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        """Informs the agent of the consequences of an action so that
        it is able to learn from it."""

        self.memory.add(state[0], action, reward[0], next_state[0], done[0])

        self.timestep += 1

        if self.timestep % self.update_interval == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self._learn(experiences, self.gamma)

        self.epsilon = max(self.eps_end, self.eps_decay * self.epsilon)

    def save(self, filename):
        """Saves the agent model's trained parameters."""

        torch.save({
            'qnetwork_local': self.qnetwork_local.state_dict(),
            'qnetwork_target': self.qnetwork_local.state_dict()},
            filename)

    def load(self, filename):
        """Loads the agent model's trained parameters."""

        state_dicts = torch.load(filename)

        self.qnetwork_local.load_state_dict(state_dicts['qnetwork_local'])
        self.qnetwork_target.load_state_dict(state_dicts['qnetwork_target'])

        self.qnetwork_local.eval()
        self.qnetwork_target.eval()

    def _learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Args:
            experiences (list): list of (s, a, r, s', done) tuples
            gamma (float): discount factor

        """
        states, actions, rewards, next_states, dones = experiences
        states = torch.as_tensor(states, dtype=torch.float)
        actions = torch.as_tensor(actions, dtype=torch.long).unsqueeze(-1)
        rewards = torch.as_tensor(rewards, dtype=torch.float).unsqueeze(-1)
        next_states = torch.as_tensor(next_states, dtype=torch.float)
        dones = torch.as_tensor(dones, dtype=torch.int8).unsqueeze(-1)

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        params_target = tau*params_local + (1 - tau)*params_target

        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
