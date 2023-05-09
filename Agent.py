import numpy as np
from Network import DeepQ
import torch as T


class Agent:
    def __init__(
        self,
        input_features,
        no_actions,
        gamma=0.99,
        epsilon=1.0,
        lr=0.001,
        replay_memory_size=100000,
        epsilon_min=0.01,
        epsilon_decrement=1e-3,
        batch_size=64,
    ):
        self.input_features = input_features
        self.no_actions = no_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.replay_memory_size = replay_memory_size
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = epsilon_decrement
        self.batch_size = batch_size
        self.can_learn = False
        self.filled_memory = 0

        self.NN = DeepQ(
            self.input_features,
            self.input_features * 2,
            self.input_features * 2,
            no_actions,
            self.lr,
        )

        self.last_memory_idx = 0
        self.state_memory = np.zeros(
            (self.replay_memory_size, input_features), dtype=np.float32
        )
        self.new_state_memory = np.zeros(
            (self.replay_memory_size, input_features), dtype=np.float32
        )
        self.chosen_action_memory = np.zeros(self.replay_memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.replay_memory_size, dtype=np.float32)
        self.is_done_memory = np.zeros(self.replay_memory_size, dtype=bool)

    def save_state_action(self, state, action, new_state, reward, is_done):
        idx = self.last_memory_idx
        self.state_memory[idx] = state
        self.new_state_memory[idx] = new_state
        self.chosen_action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.is_done_memory[idx] = is_done

        self.last_memory_idx = (idx + 1) % self.replay_memory_size
        self.can_learn = self.can_learn or self.last_memory_idx >= self.batch_size
        self.filled_memory = (
            self.filled_memory + 1
            if self.filled_memory < self.replay_memory_size
            else self.filled_memory
        )

    def predict(self, state):
        if np.random.random() > self.epsilon:
            actions = self.NN.forward(T.tensor(state))
            best_action = T.argmax(actions).item()
            return best_action
        return np.random.randint(self.no_actions)

    def sample_batch(self):
        batch = np.random.choice(self.filled_memory, self.batch_size, replace=False)

        state_batch = T.tensor(self.state_memory[batch])
        new_state_batch = T.tensor(self.new_state_memory[batch])
        chosen_action_batch = T.tensor(self.chosen_action_memory[batch], dtype=T.int64)
        reward_batch = T.tensor(self.reward_memory[batch])
        is_done_batch = T.tensor(self.is_done_memory[batch])
        return (
            state_batch,
            new_state_batch,
            chosen_action_batch,
            reward_batch,
            is_done_batch,
        )

    def learn(self):
        if not self.can_learn:
            return

        self.NN.optimizer.zero_grad()

        (
            state_batch,
            new_state_batch,
            chosen_action_batch,
            reward_batch,
            is_done_batch,
        ) = self.sample_batch()

        actions_values = self.NN.forward(state_batch)[
            np.arange(self.batch_size, dtype=np.int64), chosen_action_batch
        ]
        actions_from_new_state = self.NN.forward(new_state_batch)
        actions_from_new_state[is_done_batch] = 0.0

        expected_value = (
            reward_batch + self.gamma * T.max(actions_from_new_state, dim=1)[0]
        )

        loss = self.NN.loss(expected_value, actions_values)
        loss.backward()
        self.NN.optimizer.step()

        self.decrease_epsilon()

    def decrease_epsilon(self):
        self.epsilon = (
            self.epsilon - self.epsilon_decrement
            if self.epsilon > self.epsilon_min
            else self.epsilon_min
        )
