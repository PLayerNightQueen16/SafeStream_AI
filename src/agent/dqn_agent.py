import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


# 🧠 Neural Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.layers(x)


# 🤖 Agent
class DQNAgent:
    def __init__(self, action_space, state_size):
        self.action_space = action_space
        self.state_size = state_size

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.lr = 0.001

        self.memory = []

        self.model = DQN(state_size, len(action_space))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    # 🎯 Action selection
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)

        action_index = torch.argmax(q_values).item()
        return self.action_space[action_index]

    # 💾 Store experience
    def remember(self, state, action, reward, next_state, done):
        action_index = self.action_space.index(action)
        self.memory.append((state, action_index, reward, next_state, done))

    # 🧠 Learning step
    def learn(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state) if next_state is not None else None

            target = reward

            if not done and next_state is not None:
                target += self.gamma * torch.max(self.model(next_state)).item()

            target_f = self.model(state)
            target_f[action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        # 🔻 Reduce randomness over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay