import numpy as np
from app.models.toxicity_model import predict_toxicity


class ModerationEnv:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def reset(self):
        self.index = 0
        return self._get_state()

    def step(self, action):
        text, true_label = self.data[self.index]

        reward = self.get_reward(action, true_label)

        self.index += 1
        done = self.index >= len(self.data)

        next_state = None if done else self._get_state()

        return next_state, reward, done

    # 🔥 NEW: Convert text → state vector
    def _get_state(self):
        text, _ = self.data[self.index]

        ai_scores = predict_toxicity(text)

        state = np.array([
            ai_scores.get("toxicity", 0.0),
            ai_scores.get("insult", 0.0),
            ai_scores.get("threat", 0.0),
            ai_scores.get("obscene", 0.0),
        ])

        return state

    # 🔥 IMPROVED REWARD FUNCTION
    def get_reward(self, action, true_label):
        """
        action: 0=allow, 1=flag, 2=remove
        true_label: "safe", "flag", "remove"
        """

        action_map = ["allow", "flag", "remove"]
        predicted = action

        # ✅ Perfect decision
        if predicted == true_label:
            return 3

        # ⚠️ Slight mistake
        if predicted == "flag" and true_label in ["allow", "remove"]:
            return 1

        # ❌ Dangerous mistakes
        if predicted == "allow" and true_label == "remove":
            return -4

        if predicted == "remove" and true_label == "allow":
            return -3

        return -1