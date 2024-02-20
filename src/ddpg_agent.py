import os
from typing import Dict, Iterable

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.autograd import Variable
from torch.distributions import Normal

from src.networks import FCNetwork
from src.replay import Transition


class DiagGaussian(torch.nn.Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        eps = Variable(torch.randn(*self.mean.size()))
        return self.mean + self.std * eps


class DDPGAgent:
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        gamma: float,
        critic_learning_rate: float,
        policy_learning_rate: float,
        critic_hidden_size: Iterable[int],
        policy_hidden_size: Iterable[int],
        tau: float,
        **kwargs,
    ):
        """
        :param observation_space (gym.Space): environment's observation space
        :param action_space (gym.Space): environment's action space
        :param gamma (float): discount rate gamma
        :param critic_learning_rate (float): learning rate for critic optimisation
        :param policy_learning_rate (float): learning rate for policy optimisation
        :param critic_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected critic
        :param policy_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected policy
        :param tau (float): step for the update of the target networks
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.upper_action_bound = action_space.high[0]
        self.lower_action_bound = action_space.low[0]
        self.saveables = {}

        ACTION_SIZE = action_space.shape[0]
        if isinstance(observation_space, gym.spaces.dict.Dict):
            STATE_SIZE = 0
            for k, v in observation_space.items():
                STATE_SIZE += v.shape[0]
        else:
            STATE_SIZE = observation_space.shape[0]

        print(f"STATE_SIZE: {STATE_SIZE}")
        print(f"ACTION_SIZE: {ACTION_SIZE}")

        # set up networks
        self.actor = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE),
            output_activation=torch.nn.Tanh,
        )
        self.actor_target = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE),
            output_activation=torch.nn.Tanh,
        )
        self.actor_target.hard_update(self.actor)

        self.critic = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target.hard_update(self.critic)

        # set up optimisers
        self.policy_optim = Adam(
            self.actor.parameters(), lr=policy_learning_rate, eps=1e-3
        )
        self.policy_lr_scheduler = ExponentialLR(self.policy_optim, 0.99)
        self.critic_optim = Adam(
            self.critic.parameters(), lr=critic_learning_rate, eps=1e-3
        )
        self.critic_lr_scheduler = ExponentialLR(self.critic_optim, 0.99)

        # define loss function
        self.loss_fn = torch.nn.MSELoss()

        # define any hyperparams
        self.gamma = gamma
        self.critic_learning_rate = critic_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.tau = tau
        self.hyperparam_update_interval = 1000
        self.last_hyperparam_update = 0

        # define a gaussian for exploration
        mean = torch.zeros(ACTION_SIZE)
        std = 0.1 * torch.ones(ACTION_SIZE)
        self.noise = DiagGaussian(mean, std)

        self.saveables.update(
            {
                "actor": self.actor,
                "actor_target": self.actor_target,
                "critic": self.critic,
                "critic_target": self.critic_target,
                "policy_optim": self.policy_optim,
                "critic_optim": self.critic_optim,
            }
        )

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, filename: str, dir_path: str = ""):
        """Restores PyTorch models from models file given by path

        :param filename (str): filename containing saved models
        :param dir_path (str, optional): path to directory where models file is located
        """

        if not dir_path:
            dir_path, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dir_path, filename)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        noise = self.noise.sample() if explore else 0.0
        action = self.actor(torch.tensor(obs, dtype=torch.float32)).squeeze(0) + noise
        return (
            torch.clamp(action, self.lower_action_bound, self.upper_action_bound)
            .detach()
            .numpy()
        )

    def _cat(self, states, actions):
        return torch.cat((states, actions), dim=1)

    def compute_critic_loss(self, batch: Transition) -> torch.Tensor:
        q = self.critic(self._cat(batch.states, batch.actions))
        q_next = self.critic_target(
            self._cat(batch.next_states, self.actor_target(batch.next_states))
        )
        q_target = batch.rewards + (self.gamma * (1 - batch.done) * q_next)
        return self.loss_fn(q, q_target)

    def compute_actor_loss(self, batch: Transition) -> torch.Tensor:
        q_vals = self.critic(self._cat(batch.states, self.actor(batch.states)))
        return -q_vals.mean()

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DDPG

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        # update critic
        self.critic_optim.zero_grad()
        q_loss = self.compute_critic_loss(batch)
        q_loss.backward()
        self.critic_optim.step()

        # update actor
        self.policy_optim.zero_grad()
        p_loss = self.compute_actor_loss(batch)
        p_loss.backward()
        self.policy_optim.step()

        # soft update target networks
        self.critic_target.soft_update(self.critic, self.tau)
        self.actor_target.soft_update(self.actor, self.tau)

        return {
            "q_loss": q_loss,
            "p_loss": p_loss,
        }

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        if timestep - self.last_hyperparam_update < self.hyperparam_update_interval:
            return

        self.policy_lr_scheduler.step()
        self.critic_lr_scheduler.step()
        self.noise.std = 0.1 * (1 - timestep / max_timesteps)
        self.last_hyperparam_update = timestep
