import os
import random
from collections import deque
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ubikagent.agent.abc import Agent
from ubikagent.noise import OUNoise
from ubikagent.buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=128, fc_units2=64):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units2)
        self.fc3 = nn.Linear(fc_units2, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(
            self,
            state_size,
            action_size,
            seed,
            fcs1_units=64,
            fc2_units=32,
            fc3_units=32):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DDPGAgent(Agent):
    """Deep Deterministic Policy Gradient Agent algorithm."""

    savefilename = 'checkpoint.pth'

    def __init__(
            self,
            state_size,
            action_size,
            num_agents=1,
            update_interval=1,
            update_times=1,
            lr_actor=3e-3,
            layers_actor=[128, 64],
            lr_critic=3e-3,
            layers_critic=[64, 32, 32],
            batch_size=512,
            tau=2e-1,
            gamma=0.99,
            replay_buffer_size=1e5,
            seed=42):
        """Initializes an Agent object.

        Args:
            state_size (int): required, dimension of each state
            action_size (int): required, dimension of each discrete action
            num_agents (int): required, number of agents in the simulation
            update_interval (int): how often to update the model,
                1 = every step, 2 = every other
            update_times (int): how many times to update the model at update_interval
            lr_actor (int): learning rate for the "actor" network
            lr_critic (int): learning rate for the "critic" network
            batch_size (int): batch size for training the neural network
            tau (int): soft update of target parameters
            gamma (float): dicount factor, between 0.0 and 1.0
            replay_buffer_size (int): length of learning history from which to learn
            seed (int): random seed

        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.lr_actor = lr_actor
        self.layers_actor = layers_actor
        self.lr_critic = lr_critic
        self.layers_critic = layers_critic
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.replay_buffer_size = int(replay_buffer_size)
        self.seed = random.seed(seed)

        self.actor_local = Actor(
            state_size,
            action_size,
            fc_units=layers_actor[0],
            fc_units2=layers_actor[1],
            seed=seed).to(device)
        self.actor_target = Actor(
            state_size,
            action_size,
            fc_units=layers_actor[0],
            fc_units2=layers_actor[1],
            seed=seed).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(), lr=lr_actor)

        self.critic_local = Critic(
            state_size,
            action_size,
            fcs1_units=layers_critic[0],
            fc2_units=layers_critic[1],
            fc3_units=layers_critic[2],
            seed=seed).to(device)
        self.critic_target = Critic(
            state_size,
            action_size,
            fcs1_units=layers_critic[0],
            fc2_units=layers_critic[1],
            fc3_units=layers_critic[2],
            seed=seed).to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(), lr=lr_critic)

        self.noise = OUNoise(action_size, seed)

        self.memory = ReplayBuffer(self.replay_buffer_size, batch_size, seed)

        self.timestep = 0

        self.update_interval = update_interval
        self.update_times = update_times
        self.update_counter = 0

        self._actor_losses = deque()
        self._critic_losses = deque()

    def new_episode(self):
        """Returns statistics on the previous episode."""

        self.noise.reset()

        if len(self._actor_losses) > 0:
            loss_actor = mean(self._actor_losses)
            loss_actor_max = max(self._actor_losses)
            loss_critic = mean(self._critic_losses)
            loss_critic_max = max(self._critic_losses)
        else:
            loss_actor, loss_actor_max = 0., 0.
            loss_critic, loss_critic_max = 0., 0.

        history = {
            'loss_actor': loss_actor,
            'loss_actor_max': loss_actor_max,
            'loss_critic': loss_critic,
            'loss_critic_max': loss_critic_max,
        }

        self._actor_losses = deque()
        self._critic_losses = deque()
        return history

    def step(self, state, action, reward, next_state, done):
        """Informs the agent of the consequences of an action so that
        it is able to learn from it."""

        self.timestep += 1

        if self.num_agents > 1:
            for state, action, reward, next_state, done in zip(
                    state, action, reward, next_state, done):
                self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.add(state, action, reward, next_state, done)

        if self.timestep % self.update_interval == 0:
            if len(self.memory) > self.batch_size:
                for _ in range(self.update_times):
                    experiences = self.memory.sample()
                    self._learn(experiences, self.gamma)
                    self.update_counter += 1

    def act(self, state):
        """Return action for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += self.noise.sample()
        return np.clip(action, -1, 1)

    def save(self, directory):
        """Saves the agent model's trained parameters."""

        filepath = os.path.join(directory, self.savefilename)
        torch.save({
            'actor_local': self.actor_local.state_dict(),
            'critic_local': self.critic_local.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, filepath)

    def load(self, directory):
        """Loads the agent model's trained parameters."""

        filepath = os.path.join(directory, self.savefilename)
        state_dicts = torch.load(filepath)
        self.actor_local.load_state_dict(state_dicts['actor_local'])
        self.critic_local.load_state_dict(state_dicts['critic_local'])
        self.actor_target.load_state_dict(state_dicts['actor_target'])
        self.critic_target.load_state_dict(state_dicts['critic_target'])

    def _learn(self, experiences, gamma):

        self.update_counter += 1

        states, actions, rewards, next_states, dones = experiences

        states = torch.as_tensor(states, dtype=torch.float)
        actions = torch.as_tensor(actions, dtype=torch.float)
        rewards = torch.as_tensor(rewards, dtype=torch.float).unsqueeze(-1)
        next_states = torch.as_tensor(next_states, dtype=torch.float)
        dones = torch.as_tensor(dones, dtype=torch.int8).unsqueeze(-1)

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.critic_local, self.critic_target, self.tau)
        self._soft_update(self.actor_local, self.actor_target, self.tau)

        self._actor_losses.append(actor_loss.float().item())
        self._critic_losses.append(critic_loss.float().item())

    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(
                target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)
