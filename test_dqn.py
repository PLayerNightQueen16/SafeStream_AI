import torch
from src.agent.dqn_agent import DQNAgent

agent = DQNAgent(["allow", "flag", "remove"], 4)
agent.remember([0.1,0.2,0.3,0.4], "allow", 1, [0.2,0.3,0.4,0.5], False)
agent.remember([0.1,0.2,0.3,0.4], "allow", 1, [0.2,0.3,0.4,0.5], False)

for i in range(35):
    agent.remember([0.1,0.2,0.3,0.4], "allow", 1, [0.2,0.3,0.4,0.5], False)

try:
    agent.learn(batch_size=32)
    print("DQN learn successful")
except Exception as e:
    print("DQN learn error:", e)
