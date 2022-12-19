import copy
import os
import random

import numpy as np
import torch
from gym import make
from torch import nn
from torch.optim import Adam


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAMMA = 0.99
TAU = 1e-3
INITIAL_STEPS = 1024
TRANSITIONS = 500_000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
HID_DIM = 64


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class ExperienceBuffer:
    "Buffer for DeepQNetwork"

    def __init__(self, capacity=10_000, device=DEVICE):
        self.capacity = capacity
        self.n_stored = 0
        self.next_idx = 0
        self.device = device

        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None
        self.done = None

    def is_samplable(self, replay_size):
        return replay_size <= self.n_stored

    def add(
        self,
        state: list,
        action: int,
        next_state: list,
        reward: float,
        is_done: bool,
    ):
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)

        if self.state is None:
            self.state = torch.empty(
                [self.capacity] + list(state.shape),
                dtype=torch.float32,
                device=self.device,
            )
            self.action = torch.empty(
                self.capacity, dtype=torch.long, device=self.device
            )
            self.next_state = torch.empty(
                [self.capacity] + list(state.shape),
                dtype=torch.float32,
                device=self.device,
            )
            self.reward = torch.empty(
                self.capacity, dtype=torch.float32, device=self.device
            )
            self.done = torch.empty(self.capacity, dtype=torch.long, device=self.device)
        self.state[self.next_idx] = state
        self.action[self.next_idx] = action
        self.next_state[self.next_idx] = next_state
        self.reward[self.next_idx] = reward
        self.done[self.next_idx] = is_done

        self.next_idx = (self.next_idx + 1) % self.capacity
        self.n_stored = min(self.capacity, self.n_stored + 1)

    def get_batch(self, replay_size=BATCH_SIZE):
        idxes = torch.randperm(self.n_stored)[:replay_size]
        return (
            self.state[idxes],
            self.action[idxes].view(-1, 1),
            self.next_state[idxes],
            self.reward[idxes].view(-1, 1),
            self.done[idxes].view(-1, 1),
        )


class DeepQNetworkModel(torch.nn.Module):
    "Classic DQN"

    def __init__(self, state_dim, action_dim, hid_dim=HID_DIM):
        super().__init__()
        self.hid_dim = hid_dim
        self.activation = torch.nn.ReLU()
        self.fc1 = nn.Linear(state_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, action_dim)

    def forward(self, state):
        h = self.activation(self.fc1(state))
        h = self.activation(self.fc2(h))
        out = self.fc3(h)
        return out


class DQN:
    def __init__(self, state_dim, action_dim, hid_dim=64):
        self.steps = 0  # Do not change
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hid_dim = hid_dim
        self._buffer = ExperienceBuffer(10**5)
        self.local_model = DeepQNetworkModel(state_dim, action_dim, hid_dim).to(DEVICE)
        self.target_model = DeepQNetworkModel(state_dim, action_dim, hid_dim).to(DEVICE)
        self.target_model.eval()
        self.optimizer = Adam(self.local_model.parameters())
        self.criterion = nn.MSELoss()

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self._buffer.add(*transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        return self._buffer.get_batch()

    def train_step(self, batch):
        # Use batch to update DQN's network.
        states, actions, next_states, rewards, dones = batch

        q_pred = self.local_model(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_model(next_states).max(1)[0].unsqueeze(1)
        q_target = rewards + GAMMA * q_next * (1 - dones)

        self.optimizer.zero_grad()
        loss = self.criterion(q_pred, q_target)
        loss.backward()
        self.optimizer.step()

        self._soft_update_target_network()

    def _soft_update_target_network(self):
        for target_param, local_param in zip(
            self.target_model.parameters(), self.local_model.parameters()
        ):
            target_param.data.copy_(
                TAU * local_param.data + (1.0 - TAU) * target_param.data
            )

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self.target_model = copy.deepcopy(self.local_model)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        self.local_model.eval()
        with torch.no_grad():
            action = np.argmax(self.local_model(state).cpu().numpy())
        self.local_model.train()

        return action

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.local_model.state_dict(), "agent.pth")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.0

        while not done:
            state, reward, done, *_ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    set_seed(SEED)
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()

    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, *_ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    best_avg_rewards = -np.inf
    # pbar = tqdm(total=TRANSITIONS)
    for i in range(TRANSITIONS):
        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, *_ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        # pbar.update(1)

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            avg_reward = np.mean(rewards)
            # pbar.set_description(
            #     f"Best reward mean: {best_avg_rewards}, Reward mean: {avg_reward}, Reward std: {np.std(rewards)}"
            # )
            print(
                f"Step: {i + 1}/{TRANSITIONS}, Best reward mean: {best_avg_rewards:.2f}, Reward mean: {avg_reward:.2f}, Reward std: {np.std(rewards):.2f}"
            )
            if avg_reward > best_avg_rewards:
                best_avg_rewards = avg_reward
                dqn.save()
