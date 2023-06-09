import torch as T
import gym
from Agent import Agent

env = gym.make("LunarLander-v2")
agent = Agent(8, 4)
agent.actor_critic.load_state_dict(T.load("model_policy_value"))
epochs = 10000
scores = []
for i in range(epochs):
    score = 0
    done = False
    state = env.reset()[0]
    while not done:
        action = agent.predict(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        agent.save_state_action(state, action, reward)
        state = new_state
    agent.learn()
    scores.append(score)
    if i % 100 == 0:
        print(i, ": ", sum(scores) / len(scores))
        scores = []
        print(i, ": ", score)
    if i % 1000 == 0:
        T.save(agent.actor_critic.state_dict(), "model_policy_value")


T.save(agent.actor_critic.state_dict(), "model_policy_value")
