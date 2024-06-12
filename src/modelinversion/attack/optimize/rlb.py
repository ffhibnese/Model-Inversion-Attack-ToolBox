import os
import importlib
import random
from copy import deepcopy
from abc import ABC, abstractmethod
from collections import OrderedDict, Counter, deque
from dataclasses import dataclass, field
from typing import Tuple, Callable, Optional, Iterable
from functools import reduce


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import Tensor, LongTensor
from torch.optim import Optimizer, Adam
from torch.distributions import Normal, MultivariateNormal
from tqdm import tqdm

from ...utils import ClassificationLoss, BaseConstraint, DictAccumulator
from ...models import BaseImageClassifier, BaseImageGenerator
from ...scores import BaseLatentScore
from .base import (
    BaseImageOptimizationConfig,
    BaseImageOptimization,
    ImageOptimizationOutput,
)


GAMMA = 0.99
TAU = 1e-2
HIDDEN_SIZE = 256
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4
FIXED_ALPHA = None


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(
        self,
        state_size,
        action_size,
        device,
        hidden_size=32,
        init_w=3e-3,
        log_std_min=-20,
        log_std_max=2,
    ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.device = device
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)
        # self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mu.weight.data.uniform_(*hidden_init(self.mu))
        self.log_std_linear.weight.data.uniform_(*hidden_init(self.log_std_linear))

    def forward(self, state):

        x = F.relu(self.fc1(state), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(self.device)
        action = torch.tanh(mu + e * std)
        log_prob = (
            Normal(mu, std).log_prob(mu + e * std)
            - torch.log(1 - action.pow(2) + epsilon)
        ).mean(1, keepdim=True)

        return action, log_prob

    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        # state = torch.FloatTensor(state).to(self.device) #.unsqzeeze(0)
        mu, log_std = self(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(self.device)
        action = torch.tanh(mu + e * std)
        # action = torch.clamp(action*action_high, action_low, action_high)
        return action[0]


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, device, hidden_size=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.device = device
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        # self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class Experience:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(
        self, state_size, action_size, device, hidden_size, action_prior="uniform"
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        # self.seed = random.seed(random_seed)
        self.device = device

        self.target_entropy = -action_size  # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], device=device, requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=LR_ACTOR)
        self._action_prior = action_prior

        # print("Using: ", device)

        # Actor Network
        self.actor_local = Actor(state_size, action_size, device, hidden_size).to(
            self.device
        )
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=LR_ACTOR, weight_decay=1e-4
        )

        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, device, hidden_size).to(
            self.device
        )
        self.critic2 = Critic(state_size, action_size, device, hidden_size).to(
            self.device
        )

        self.critic1_target = Critic(state_size, action_size, device, hidden_size).to(
            self.device
        )
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, device, hidden_size).to(
            self.device
        )
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=LR_CRITIC, weight_decay=1e-4
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=LR_CRITIC, weight_decay=1e-4
        )

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, device)

    def step(self, state, action, reward, next_state, done, step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(step, experiences, GAMMA)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        # state = torch.from_numpy(state).float().to(self.device)
        # state = state.to(self.device)
        action = self.actor_local.get_action(state)
        return action.detach()

    def learn(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_action, log_pis_next = self.actor_local.evaluate(next_states)

        Q_target1_next = self.critic1_target(next_states, next_action.squeeze(0))
        Q_target2_next = self.critic2_target(next_states, next_action.squeeze(0))

        # take the mean of both critics for updating
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)

        if FIXED_ALPHA == None:
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (
                gamma
                * (1 - dones)
                * (Q_target_next - self.alpha * log_pis_next.squeeze(0))
            )
        else:
            Q_targets = rewards + (
                gamma
                * (1 - dones)
                * (Q_target_next - FIXED_ALPHA * log_pis_next.squeeze(0))
            )

        # Compute critic loss
        Q_1 = self.critic1(states.detach(), actions.detach())
        Q_2 = self.critic2(states.detach(), actions.detach())

        # print(self.critic1.fc3.weight.requires_grad)
        # print(Q_1.requires_grad)
        # exit()
        critic1_loss = 0.5 * F.mse_loss(Q_1, Q_targets.detach())
        critic2_loss = 0.5 * F.mse_loss(Q_2, Q_targets.detach())
        # Update critics
        # critic 1
        # print(critic1_loss.requires_grad)
        # exit()
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        if step % d == 0:
            # ---------------------------- update actor ---------------------------- #
            if FIXED_ALPHA == None:
                alpha = torch.exp(self.log_alpha)
                # Compute alpha loss
                actions_pred, log_pis = self.actor_local.evaluate(states)
                alpha_loss = -(
                    self.log_alpha * (log_pis + self.target_entropy).detach()
                ).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = alpha
                # Compute actor loss
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(
                        loc=torch.zeros(self.action_size),
                        scale_tril=torch.ones(self.action_size).unsqueeze(0),
                    )
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0

                actor_loss = (
                    alpha * log_pis.squeeze(0)
                    - self.critic1(states, actions_pred.squeeze(0))
                    - policy_prior_log_probs
                ).mean()
            else:

                actions_pred, log_pis = self.actor_local.evaluate(states)
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(
                        loc=torch.zeros(self.action_size),
                        scale_tril=torch.ones(self.action_size).unsqueeze(0),
                    )
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0

                actor_loss = (
                    FIXED_ALPHA * log_pis.squeeze(0)
                    - self.critic1(states, actions_pred.squeeze(0))
                    - policy_prior_log_probs
                ).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1, self.critic1_target, TAU)
            self.soft_update(self.critic2, self.critic2_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


@dataclass
class RlbOptimizationConfig(BaseImageOptimizationConfig):

    train_agent_iter_times: int = 4000
    optim_steps: int = 1
    truncation: float = 0
    state_action_loss_weights: Tuple[int, int] = field(default_factory=lambda: (2, 2))


class RlbOptimization(BaseImageOptimization):

    def __init__(
        self,
        config: RlbOptimizationConfig,
        generator: BaseImageGenerator,
        state_image_loss_fn: Callable[
            [Tensor, LongTensor], Tensor | Tuple[Tensor, OrderedDict]
        ],  # ce + 4 * log maxmargin
        action_image_loss_fn: Callable[
            [Tensor, LongTensor], Tensor | Tuple[Tensor, OrderedDict]
        ],  # ce
    ) -> None:
        super().__init__(config)

        self.generator = generator
        self.state_image_loss_fn = state_image_loss_fn
        self.action_image_loss_fn = action_image_loss_fn
        self.agent_dir = os.path.join(config.experiment_dir, 'agents')
        os.makedirs(self.agent_dir, exist_ok=True)

    def _get_agent_path(self, label: int):
        return os.path.join(self.agent_dir, f'agent_{label}.pt')

    def _save_agent(self, label, agent):
        path = self._get_agent_path(label)
        torch.save(agent, path)

    def _load_agent(self, label):
        path = self._get_agent_path(label)
        if not os.path.exists(path):
            return None
        return torch.load(path)

    # @torch.no_grad()
    def _agent_training(self, label: int, input_shape: list[int], num):
        print(f'train agent for class {label}')

        config: RlbOptimizationConfig = self.config
        device = config.device
        truncation = config.truncation

        input_dim = reduce(lambda a, b: a * b, input_shape)

        y = torch.LongTensor([label]).to(device)

        agent = Agent(
            state_size=input_dim, action_size=input_dim, device=device, hidden_size=256
        )

        best_losses = 1e10
        best_zs = []
        # best_imgs = []
        best_agent = None
        for iter_time in tqdm(range(config.train_agent_iter_times), leave=True):

            z = torch.randn((1, input_dim), device=device)
            state = z.detach().reshape(1, -1)

            for t in range(config.optim_steps):

                with torch.no_grad():
                    action = agent.act(state)
                    z = truncation * z + (1 - truncation) * action
                    next_state = z.detach()

                    state_image = self.generator(z.reshape((1, *input_shape)), labels=y)
                    action_image = self.generator(
                        action.reshape((1, *input_shape)), labels=y
                    )

                    loss_state = self.state_image_loss_fn(state_image, y)
                    loss_action = self.action_image_loss_fn(action_image, y)

                    if isinstance(loss_state, Tuple):
                        loss_state = loss_state[0]
                    if isinstance(loss_action, Tuple):
                        loss_action = loss_action[0]

                    reward = (
                        -config.state_action_loss_weights[0] * loss_state.detach()
                        - config.state_action_loss_weights[1] * loss_action.detach()
                    )

                agent.step(
                    state.detach().cpu().numpy(),
                    action.detach().cpu().numpy(),
                    reward.detach().cpu().numpy(),
                    next_state.detach().cpu().numpy(),
                    t == config.optim_steps - 1,
                    t,
                )
                state = next_state

            test_images = []
            test_zs = []
            test_losses = []
            for i in range(num):
                with torch.no_grad():
                    z_test = torch.randn((1, input_dim), device=device)
                    for t in range(config.optim_steps):
                        state_test = z_test.detach()
                        action_test = agent.act(state_test)
                        z_test = truncation * z_test + (1 - truncation) * action_test
                    test_image = self.generator(z_test).detach()

                    test_loss = self.state_image_loss_fn(state_image, y)
                    if isinstance(test_loss, Tuple):
                        test_loss = test_loss[0]
                    test_images.append(test_image.cpu())
                    test_zs.append(z_test)

                test_losses.append(test_loss.item())

            if np.array(test_losses).mean() < best_losses:
                best_zs = test_zs
                # best_imgs = test_images
                # best_agent = deepcopy(agent)

                self._save_agent(label, agent)
        return torch.cat(best_zs, dim=0)

    # @torch.no_grad()
    def __call__(
        self, latents: Tensor, labels: LongTensor
    ) -> Tuple[Tensor, LongTensor]:
        targets = set(labels.tolist())

        config: RlbOptimizationConfig = self.config
        device = config.device

        input_shape = list(latents.shape[1:])
        input_dim = reduce(lambda a, b: a * b, input_shape)

        result_zs = []
        result_labels = []

        for target in targets:
            target_latents = latents[labels == target]
            # agent = self._load_agent(target)
            # if agent is None:
            result_zs.append(
                self._agent_training(target, input_shape, len(target_latents))
            )
            result_labels.extend([target] * len(latents))
            # agent = self._load_agent(target)

            # else:
            #     with torch.no_grad():

            #         for i in range(len(target_latents)):
            #             state = (
            #                 target_latents[i].detach().to(device).reshape(1, input_dim)
            #             )
            #             for t in range(config.optim_steps):
            #                 action = agent.act(state)
            #                 state = (
            #                     config.truncation * state
            #                     + (1 - config.truncation) * action
            #                 )
            #             z = state.reshape(input_shape)
            #             result_zs.append(z.detach().cpu())
            #             result_labels.append(target)

        result_zs = torch.cat(result_zs, dim=0)
        result_labels = torch.LongTensor(result_labels)
        # print(result_zs.shape, result_labels.shape)
        # exit()
        result_imgs = self.generator(result_zs, labels=result_labels)
        return ImageOptimizationOutput(
            images=result_imgs.detach().cpu(),
            labels=result_labels.detach().cpu(),
            latents=result_zs.detach().cpu(),
        )
