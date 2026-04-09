def train(env, agent, episodes=50, batch_size=32):
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        done = False

        while not done:
            # 🎯 choose action
            action = agent.choose_action(state)

            # environment step
            next_state, reward, done = env.step(action)

            # 💾 store experience
            agent.remember(state, action, reward, next_state, done)

            # 🧠 learn from memory
            agent.learn(batch_size)

            # move forward
            state = next_state
            total_reward += reward

        print(f"Episode {ep+1}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")