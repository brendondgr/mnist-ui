# Import Torch
import torch

# Optimizer Imports
from torch.optim import SGD, Adam, Adagrad, Adadelta, RMSprop

# Import Loss/Criterion
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss, NLLLoss

# Import General Other
import tqdm
import os
from training import TrainingLoop

# Model 1 #
# Layers:
# 6 fully connected layers
#   - Input layer: 784 neurons
#   - Hidden layer 1: 256 neurons
#   - Hidden layer 2: 512 neurons
#   - Hidden layer 3: 256 neurons
#   - Hidden layer 4: 128 neurons
#   - Output layer: 10 neurons
class Model_1(torch.nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.layer1 = torch.nn.Linear(784, 256)
        self.layer2 = torch.nn.Linear(256, 512)
        self.layer3 = torch.nn.Linear(512, 256)
        self.layer4 = torch.nn.Linear(256, 128)
        self.layer5 = torch.nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        return x

class Model_1_Trainer(TrainingLoop):
    def __init__(self, train_dataset, test_dataset, epochs=10, batch_size=32, loss=0.01, num_workers=4):
        super(Model_1_Trainer, self).__init__()
        self.model = Model_1()
        self.loss = loss
        self.optimizer = SGD(self.model.parameters(), lr=self.loss)
        self.loss_fn = CrossEntropyLoss()
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Data Loaders
        self.train_loader = self.dataloader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        self.test_loader = self.dataloader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        
    def run(self):
        # Run Training Loop
        self.train(self.model, self.optimizer, self.loss_fn, self.train_loader, self.epochs)
        
    def test_loaders(self):
        for images, labels in self.test_loader:
            # Print First
            print(images.shape, labels.shape)
            break