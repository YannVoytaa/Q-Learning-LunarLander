import torch as T
import gym
from Agent import Agent

env = gym.make("LunarLander-v2", render_mode="human")
agent = Agent(8, 4, lr=0.0, epsilon=0.0, epsilon_decrement=0.0, epsilon_min=0.0)
agent.NN.load_state_dict(T.load("model"))
iters = 500
for i in range(iters):
    score = 0
    done = False
    state = env.reset()[0]
    while not done:
        action = agent.predict(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        state = new_state
    print(i, ": ", score)
