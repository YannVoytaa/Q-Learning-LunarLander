import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorCriticNetwork(nn.Module):
    def __init__(
        self, input_features, hidden_features, hidden2_features, no_actions, lr=0.001
    ):
        super(ActorCriticNetwork, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.hidden2_features = hidden2_features
        self.no_actions = no_actions
        self.input_layer = nn.Linear(input_features, hidden_features)
        self.hidden_layer = nn.Linear(hidden_features, hidden2_features)
        self.policy_layer1 = nn.Linear(hidden2_features, hidden2_features)
        self.policy_layer2 = nn.Linear(hidden2_features, no_actions)
        self.value_layer1 = nn.Linear(hidden2_features, hidden2_features)
        self.value_layer2 = nn.Linear(hidden2_features, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = self.input_layer(state)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        policy = self.policy_layer1(x)
        policy = F.relu(policy)
        policy = self.policy_layer2(policy)
        policy = F.softmax(policy)
        value = self.value_layer1(x)
        value = F.relu(value)
        value = self.value_layer2(value)

        return policy, value
