import os
import random
from collections import deque
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ubikagent.agent.abc import Agent
from ubikagent.buffer import ReplayBuffer, PrioritizedReplayBuffer
from ubikagent.model import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent(Agent):
    """Deep Q-Learning algorithm for discrete actions spaces.

    Deep Q-Learning algorithm first introduced by DeepMind's [research paper]
    (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

    Q-Learning is a model-free reinforcement learning algorithm to learn a policy
    for an agent. In Deep Q-Learning, a neural network represents the Q. More
    specifically, the DQN takes the state as an input and returns the corresponding
    predicted action values for each possible actions.

    """
    savefilename = 'checkpoint.pth'

    def __init__(
            self,
            state_size,
            action_size,
            num_agents=1,
            learning_rate=5e-4,
            batch_size=64,
            tau=1e-3,
            gamma=0.99,
            update_interval=4,
            update_times=1,
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
            update_interval (int): how often to update the model,
                1 = every step, 2 = every other
            update_times (int): how many times to update the model at update_interval
            replay_buffer_size (int): length of learning history from which to learn
            seed (int): random seed
            eps_start (float): starting value of epsilon, for epsilon-greedy
                action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor for decreasing epsilon (per
                episode)

        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
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
        self.eps_at_episode_start = eps_start

        self.timestep = 0

        self.update_interval = update_interval
        self.update_times = update_times
        self.update_counter = 0

        self._loss_history = deque()

    def new_episode(self):
        """Returns statistics on the previous episode."""

        if len(self._loss_history) > 0:
            loss = mean(self._loss_history)
            loss_max = max(self._loss_history)
        else:
            loss, loss_max = 0., 0.

        history = {
            'epsilon': self.eps_at_episode_start,
            'loss': loss,
            'loss_max': loss_max
        }

        self.eps_at_episode_start = self.epsilon
        self._loss_history.clear()
        return history

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

        return random.randrange(self.action_size)

    def step(self, state, action, reward, next_state, done):
        """Informs the agent of the consequences of an action so that
        it is able to learn from it."""

        if self.num_agents > 1:
            for state, action, reward, next_state, done in zip(
                    state, action, reward, next_state, done):
                self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.add(state, action, reward, next_state, done)

        self.timestep += 1

        if self.timestep % self.update_interval == 0:
            if len(self.memory) > self.batch_size:
                for _ in range(self.update_times):
                    experiences = self.memory.sample()
                    self._learn(experiences, self.gamma)
                    self.update_counter += 1

        self.epsilon = max(self.eps_end, self.eps_decay * self.epsilon)

    def save(self, directory):
        """Saves the agent model's trained parameters."""

        filepath = os.path.join(directory, self.savefilename)
        torch.save({
            'qnetwork_local': self.qnetwork_local.state_dict(),
            'qnetwork_target': self.qnetwork_local.state_dict()},
            filepath)

    def load(self, directory):
        """Loads the agent model's trained parameters."""

        filepath = os.path.join(directory, self.savefilename)
        state_dicts = torch.load(filepath)

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

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)  # noqa: E501
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        self._loss_history.append(loss.float().item())

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        params_target = tau*params_local + (1 - tau)*params_target

        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
                target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)


class DQNAgentWithPER(DQNAgent):
    """The class extends DQNAgent by using Prioritized Experience Replay."""

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

        self.memory = PrioritizedReplayBuffer(
            self.replay_buffer_size, self.batch_size, seed=self.seed)
        self._td_errors = np.array([])
        self._is_weights = np.array([])

    def new_episode(self):
        """Returns statistics on the previous episode."""

        history = {
            'epsilon': self.epsilon,
            'loss': self._loss_history,
            'td_error': self._td_errors,
            'is_weights': self._is_weights,
        }

        self._loss_history.clear()
        self._td_errors = np.array([])
        self._is_weights = np.array([])
        return history

    def step(self, state, action, reward, next_state, done):
        """Informs the agent of the consequences of an action so that
        it is able to learn from it."""

        if self.num_agents > 1:
            for state, action, reward, next_state, done in zip(
                    state, action, reward, next_state, done):
                self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.add(state, action, reward, next_state, done)
        self.timestep += 1

        if self.timestep % self.update_interval == 0:
            if len(self.memory) > self.batch_size:
                for _ in range(self.update_times):
                    experiences = self.memory.sample()
                    td_errors = self._learn(experiences, self.gamma)
                    self.memory.update_priorities(td_errors)
                    self.update_counter += 1

        self.epsilon = max(self.eps_end, self.eps_decay * self.epsilon)

    def _learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Args:
            experiences (list): list of (s, a, r, s', done) tuples
            gamma (float): discount factor

        """
        weights, states, actions, rewards, next_states, dones = experiences
        is_weights = torch.as_tensor(weights, dtype=torch.float).unsqueeze(-1)
        states = torch.as_tensor(states, dtype=torch.float)
        actions = torch.as_tensor(actions, dtype=torch.long).unsqueeze(-1)
        rewards = torch.as_tensor(rewards, dtype=torch.float).unsqueeze(-1)
        next_states = torch.as_tensor(next_states, dtype=torch.float)
        dones = torch.as_tensor(dones, dtype=torch.int8).unsqueeze(-1)

        Q_targets_next = self.qnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = is_weights * (rewards + (gamma * Q_targets_next * (1 - dones)))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        self._loss_history.append(loss.float().item())

        td_errors = Q_targets.detach().numpy() - Q_expected.detach().numpy()
        td_errors = np.abs(td_errors)
        self._td_errors = np.append(self._td_errors, td_errors)
        self._is_weights = np.append(self._is_weights, weights)

        return td_errors.reshape(-1).tolist()
