import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, num_actions1, num_actions2):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head1 = nn.Linear(384, num_actions1)
        self.head2 = nn.Linear(384, num_actions2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head1(x.view(x.size(0), -1)),self.head2(x.view(x.size(0), -1))
