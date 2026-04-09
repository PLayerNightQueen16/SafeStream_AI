from src.env.moderation_env import ModerationEnv
from src.agent.dqn_agent import DQNAgent
from data.samples.sample_data import data
from src.training.train_rl import train

# Create environment
env = ModerationEnv(data)

# Define actions
actions = ["allow", "flag", "remove"]

# Create agent
agent = DQNAgent(actions)

# Train
train(env, agent, episodes=20)