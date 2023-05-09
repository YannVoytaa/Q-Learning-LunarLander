import torch as T
import gym
from Agent import Agent

env = gym.make("LunarLander-v2")
agent = Agent(8, 4)
agent.NN.load_state_dict(T.load("model"))
epochs = 500
for i in range(epochs):
    score = 0
    done = False
    state = env.reset()[0]
    while not done:
        action = agent.predict(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        agent.save_state_action(state, action, new_state, reward, done)
        agent.learn()
        state = new_state
    if i % 100 == 0:
        print(i, ": ", score)

T.save(agent.NN.state_dict(), "model")
