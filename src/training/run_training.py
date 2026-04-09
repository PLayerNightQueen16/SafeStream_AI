from src.env.moderation_env import ModerationEnv
from src.agent.dqn_agent import DQNAgent
from src.training.train_rl import train

# 🧪 small dataset
data = [
    ("I love this", "allow"),
    ("you are stupid", "flag"),
    ("I will kill you", "remove"),
    ("this is garbage", "flag"),
    ("great job!", "allow"),
]

env = ModerationEnv(data)

agent = DQNAgent(
    action_space=["allow", "flag", "remove"],
    state_size=4
)

# 🔥 train
train(env, agent, episodes=100)

# 💾 save model
import torch
torch.save(agent.model.state_dict(), "dqn_model.pth")

print("✅ Training complete + model saved!")