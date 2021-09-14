import abc
import copy
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.functional as torch_func
import torch.optim as optim

from deep_rl.model import QNetwork, Actor, Critic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(
        target_model.parameters(), local_model.parameters()
    ):
        target_param.data.copy_(
            tau * local_param.data + (1.0 - tau) * target_param.data
        )


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
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
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = seed
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, action_dtype=torch.long):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(DEVICE)
        )
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).to(dtype=action_dtype, device=DEVICE)
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(DEVICE)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(DEVICE)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(DEVICE)
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Agent(abc.ABC):
    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done) -> None:
        pass

    @abc.abstractmethod
    def act(self, state, eps=0.0) -> int:
        pass

    @abc.abstractmethod
    def learn(self, experiences, gamma) -> None:
        pass


class DQNAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size,
        action_size,
        seed=0,
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.99,
        tau=1e-3,
        learning_rate=5e-4,
        update_lag=4,
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.update_lag = update_lag
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        self.memory = ReplayBuffer(action_size, self.buffer_size, batch_size, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_lag
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = (
            self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        )
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = torch_func.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = seed
        random.seed(seed)
        self.state = None
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for i in range(len(x))]
        )
        self.state = x + dx
        return self.state


class DDPGAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size,
        action_size,
        seed=0,
        buffer_size=int(1e5),
        batch_size=128,
        gamma=0.99,
        tau=1e-3,
        actor_learning_rate=2e-4,
        critic_learning_rate=2e-4,
        weight_decay=0,
        update_lag=20,
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        self.update_lag = update_lag
        self.t_step = 0

        self.actor_local = Actor(state_size, action_size, seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=actor_learning_rate
        )

        self.critic_local = Critic(state_size, action_size, seed).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, seed).to(DEVICE)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=critic_learning_rate,
            weight_decay=weight_decay,
        )

        self.noise = OUNoise(action_size, seed)
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        """
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        self._replay()

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        # FIXME: action range shouldn't be hard coded here
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        self._update_critic(actions, dones, gamma, next_states, rewards, states)
        self._update_actor(states)

        soft_update(self.critic_local, self.critic_target, self.tau)
        soft_update(self.actor_local, self.actor_target, self.tau)

    def _update_actor(self, states):
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def _update_critic(self, actions, dones, gamma, next_states, rewards, states):
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def _replay(self):
        self.t_step = (self.t_step + 1) % self.update_lag
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(action_dtype=torch.float32)
                self.learn(experiences, self.gamma)
