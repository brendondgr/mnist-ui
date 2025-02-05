
# Import for Torch Modules that Can be Used to Create CNNs and NNs
import torch.nn as nn
import torch.nn.functional as F

class MediumModel(nn.Module):
    def __init__(self):
        super(MediumModel, self).__init__()
        # Conv1
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)
        self.linear1 = nn.Linear(10*14*14, 324)
        self.conv2 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        self.linear2 = nn.Linear(10*9*9, 64)
        self.conv3 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.linear3 = nn.Linear(5*4*4, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = x.view(-1, 1, 18, 18)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear2(x))
        x = x.view(-1, 1, 8, 8)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.linear3(x)
        return x