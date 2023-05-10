import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(
        self, input_features, hidden_features, hidden2_features, no_actions, lr=0.001
    ):
        super(PolicyNetwork, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.hidden2_features = hidden2_features
        self.no_actions = no_actions
        self.input_layer = nn.Linear(input_features, hidden_features)
        self.hidden_layer = nn.Linear(hidden_features, hidden2_features)
        self.hidden_layer2 = nn.Linear(hidden2_features, no_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = self.input_layer(state)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.hidden_layer2(x)
        x = F.softmax(x)

        return x
