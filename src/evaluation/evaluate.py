def evaluate(agent, env):
    correct = 0
    total = len(env.data)

    for text, label in env.data:
        action = agent.choose_action(text)
        if action == label:
            correct += 1

    print("Accuracy:", correct / total)