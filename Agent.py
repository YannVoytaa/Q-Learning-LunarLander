import numpy as np
from ActorCriticNetwork import ActorCriticNetwork
import torch as T


class Agent:
    def __init__(
        self,
        input_features,
        no_actions,
        gamma=0.99,
        lr=0.001,
    ):
        self.input_features = input_features
        self.no_actions = no_actions
        self.gamma = gamma
        self.lr = lr
        hidden_features = self.input_features**2
        hidden_features2 = self.input_features**2

        self.state_memory = []
        self.chosen_action_memory = []
        self.reward_memory = []

        self.actor_critic = ActorCriticNetwork(
            input_features, hidden_features, hidden_features2, no_actions, lr
        )

    def save_state_action(self, state, action, reward):
        self.state_memory.append(state)
        self.chosen_action_memory.append(action)
        self.reward_memory.append(reward)

    def predict(self, state):
        probs, _ = self.actor_critic.forward(T.tensor(state))
        action = probs.multinomial(num_samples=1).item()
        return action

    def learn(self):
        state_memory = np.array(self.state_memory)
        chosen_action_memory = np.array(self.chosen_action_memory)
        reward_memory = np.array(self.reward_memory)
        filled_memory = len(reward_memory)
        idxs = np.arange(filled_memory)

        probs, values = self.actor_critic.forward(T.tensor(state_memory))
        probs = probs[idxs, chosen_action_memory]
        G = np.zeros_like(reward_memory)
        policy_loss = np.zeros_like(reward_memory)
        value_loss = np.zeros_like(reward_memory)
        for t in range(filled_memory - 2, -1, -1):
            G[t] = reward_memory[t] + self.gamma * G[t + 1]
        G = T.tensor(G)
        advantages = G - values.view(-1)
        policy_loss = -T.log(probs.clamp(min=1e-8, max=1 - 1e-8)).view(-1) * advantages
        value_loss = advantages**2

        loss = T.mean(policy_loss + value_loss)

        self.actor_critic.zero_grad()
        loss.backward()
        self.actor_critic.optimizer.step()

        # Clear memory lists
        self.state_memory.clear()
        self.chosen_action_memory.clear()
        self.reward_memory.clear()
