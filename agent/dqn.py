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
    """Interacts with and learns from the environment."""

    def __init__(
            self,
            state_size,
            action_size,
            learning_rate=5e-4,
            batch_size=64,
            tau=1e-3,
            gamma=0.99,
            update_interval=4,
            replay_buffer_size=1e5,
            seed=42):
        """Initializes an Agent object.

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            learning_rate (int): learning rate for the neural network
            batch_size (int): batch size for training the neural network
            tau (int): soft update of target parameters
            gamma (float): dicount factor, between 0.0 and 1.0
            update_interval (int): how often to update the network
            replay_buffer_size (int): length of learning history from which to learn
            seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
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
        self.timestep = 0

    def act(self, state, eps=0.):
        """Returns action for given state as per current policy.

        Uses epsilon-greedy action selection. When epsilon is 1.0,
        the action is totally random. When epsilon is 0.0 the returned
        action is always the best action according the current policy.

        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection

        Returns:
            int: chosen action
        """

        if random.random() > eps:

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

        self.memory.add(state, action, reward, next_state, done)

        self.timestep = (self.timestep + 1) % self.update_interval
        if self.timestep == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self._learn(experiences, self.gamma)

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
