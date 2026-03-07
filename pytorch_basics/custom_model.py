import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 4)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(4, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x