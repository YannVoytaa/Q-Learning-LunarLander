import numpy as np
from PolicyNetwork import PolicyNetwork
import torch as T


class Agent:
    def __init__(
        self,
        input_features,
        no_actions,
        gamma=0.99,
        lr=0.001,
        replay_memory_size=100000,
    ):
        self.input_features = input_features
        self.no_actions = no_actions
        self.gamma = gamma
        self.lr = lr
        self.replay_memory_size = replay_memory_size
        self.filled_memory = 0
        self.actions = [i for i in range(no_actions)]
        hidden_features = self.input_features**2
        hidden_features2 = self.input_features**2

        self.state_memory = np.zeros(
            (self.replay_memory_size, input_features), dtype=np.float32
        )
        self.chosen_action_memory = np.zeros(self.replay_memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.replay_memory_size, dtype=np.float32)

        self.policy = PolicyNetwork(
            input_features, hidden_features, hidden_features2, no_actions, lr
        )

    def save_state_action(self, state, action, reward):
        if self.filled_memory >= self.replay_memory_size:
            return
        idx = self.filled_memory
        self.state_memory[idx] = state
        self.chosen_action_memory[idx] = action
        self.reward_memory[idx] = reward

        self.filled_memory += 1

    def predict(self, state):
        probs = self.policy.forward(T.tensor(state, requires_grad=False))
        action = probs.multinomial(num_samples=1).item()
        return action

    def learn(self):
        actions = np.zeros([self.filled_memory, self.no_actions])
        idxs = np.arange(self.filled_memory)
        actions[idxs, self.chosen_action_memory[idxs]] = 1

        # discounts = self.gamma ** np.arange(self.filled_memory)
        # returns = np.cumsum(
        #    self.reward_memory[: self.filled_memory] * discounts, axis=0
        # )
        # G = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        G = np.zeros_like(self.reward_memory[idxs])
        for t in range(self.filled_memory):
            G_sum = 0
            discount = 1
            for k in range(t, self.filled_memory):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma

            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        self.policy.zero_grad()
        y_pred = self.policy.forward(T.tensor(self.state_memory[idxs]))
        y_true = T.tensor(actions[idxs])
        loss = -T.mean(
            y_true
            * T.log(T.clip(y_pred, 1e-8, 1 - 1e-8))
            * T.tensor(G).view(G.shape[0], 1)
        )

        loss.backward()
        self.policy.optimizer.step()

        self.filled_memory = 0
