import abc
from collections import namedtuple, deque
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(abc.ABC):

    @abc.abstractmethod
    def act(self, state, eps=0.):
        pass

    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abc.abstractmethod
    def load(self, filename):
        pass

    @abc.abstractmethod
    def save(self, filename):
        pass

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

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        self.memory = ReplayBuffer(action_size, self.replay_buffer_size, batch_size, seed)
        self.timestep = 0

    def act(self, state, eps=0.):
        """Returns action for given state as per current policy.

        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection

        Returns:
            int: chosen action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
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
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

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


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Args:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Adds a new experience to memory."""

        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self):
        """Randomly samples a batch of experiences from memory."""

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack(
            [exp.state for exp in experiences if exp is not None]
        )).float().to(device)
        actions = torch.from_numpy(np.vstack(
            [exp.action for exp in experiences if exp is not None]
        )).long().to(device)
        rewards = torch.from_numpy(np.vstack(
            [exp.reward for exp in experiences if exp is not None]
        )).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [exp.next_state for exp in experiences if exp is not None]
        )).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [exp.done for exp in experiences if exp is not None]
        ).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
