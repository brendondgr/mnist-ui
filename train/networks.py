# Import for Torch Modules that Can be Used to Create CNNs and NNs
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkWrapper:
    def __init__(self, network_name, print_info=False):
        # The Network Name
        self.network_name = network_name
        
        # Model
        self.model = self.get_network()
        
        # Print Info if True
        if print_info: self.model_printer()
        
    def get_network(self):
        # If the Name is "SimpleModel", return a SimpleModel
        if self.network_name == "SimpleModel":
            return SimpleModel()
        elif self.network_name == "MediumModel":
            return MediumModel()
        else:
            raise ValueError(f"Network {self.network_name} not found.")
    
    def model_printer(self):
        # Get Number of Trainable Parameters in the Model...
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Print Model Name and Number of Trainable Parameters
        print(f'#############################################')
        print(f'## NETWORK INFORMATION ##')
        print(f"Model: {self.network_name}")
        print(f"Number of Trainable Parameters: {num_params}")
        print(f'#############################################\n')


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, kernel_size=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(25, 3, kernel_size=5, padding=1)
        self.linear = nn.Linear(3*13*13, 10)
        self.first_time = True
        
    def forward(self, x):
        if self.first_time: print(f"#############################################")
        if self.first_time: print(f"## MODEL SHAPE PER LAYER ##")
        if self.first_time: print(f"Input: {x.shape}")
        x = F.relu(self.conv1(x))
        if self.first_time: print(f"Conv1: {x.shape}")
        x = self.pool(x)
        if self.first_time: print(f"Pool1: {x.shape}")
        x = F.relu(self.conv2(x))
        if self.first_time: print(f"Conv2: {x.shape}")
        x = x.view(-1, 3*13*13)
        if self.first_time: print(f"View: {x.shape}")
        x = self.linear(x)
        if self.first_time: print(f"Linear: {x.shape}")
        if self.first_time: print(f'#############################################\n')
        self.first_time = False
        return x

class MediumModel(nn.Module):
    def __init__(self):
        super(MediumModel, self).__init__()
        # Conv1
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)
        # Linear to 26x26
        self.linear1 = nn.Linear(10*14*14, 324)
        # Conv2
        self.conv2 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        # Linear to 10x10
        self.linear2 = nn.Linear(10*9*9, 64)
        self.conv3 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        # Linear from 5x5x5 to 10...
        self.linear3 = nn.Linear(5*4*4, 10)
        self.first_time = True

    def forward(self, x):
        if self.first_time: print(f"#############################################")
        if self.first_time: print(f"## MODEL SHAPE PER LAYER ##")
        if self.first_time: print(f"Input: {x.shape}")
        x = F.relu(self.conv1(x))
        if self.first_time: print(f"Conv1: {x.shape}")
        x = self.pool1(x)
        if self.first_time: print(f"Pool1: {x.shape}")
        x = self.dropout1(x)
        if self.first_time: print(f"Dropout1: {x.shape}")
        x = x.view(x.size(0), -1)
        if self.first_time: print(f"View1: {x.shape}")
        x = F.relu(self.linear1(x))
        if self.first_time: print(f"Linear1: {x.shape}")
        x = x.view(-1, 1, 18, 18)
        if self.first_time: print(f"View2: {x.shape}")
        x = F.relu(self.conv2(x))
        if self.first_time: print(f"Conv2: {x.shape}")
        x = self.pool2(x)
        if self.first_time: print(f"Pool2: {x.shape}")
        x = self.dropout2(x)
        if self.first_time: print(f"Dropout2: {x.shape}")
        x = x.view(x.size(0), -1)
        if self.first_time: print(f"View3: {x.shape}")
        x = F.relu(self.linear2(x))
        if self.first_time: print(f"Linear2: {x.shape}")
        x = x.view(-1, 1, 8, 8)
        if self.first_time: print(f"View4: {x.shape}")
        x = F.relu(self.conv3(x))
        if self.first_time: print(f"Conv3: {x.shape}")
        x = self.pool3(x)
        if self.first_time: print(f"Pool3: {x.shape}")
        x = x.view(x.size(0), -1)
        if self.first_time: print(f"View5: {x.shape}")
        x = self.linear3(x)
        if self.first_time: print(f"Linear3: {x.shape}")
        if self.first_time: print(f'#############################################\n')
        self.first_time = False
        return x


if __name__ == "__main__":
    # Wrapper
    wrapper = NetworkWrapper("SimpleModel")