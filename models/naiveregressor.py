import torch.nn as nn
import torch.nn.functional as F

class NaiveRegressor(nn.Module):
    def __init__(self):
        super(NaiveRegressor, self).__init__()
        self.conv1 = nn.Conv1d(2, 4, 11)
        self.conv2 = nn.Conv1d(4, 16, 17)
        self.fc = nn.Linear(6, 1024)
        self.conv3 = nn.Conv1d(16, 2, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 8)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 16)
        x = self.conv3(self.fc(x))
        return x


