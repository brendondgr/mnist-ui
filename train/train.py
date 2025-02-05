# General Imports
import struct
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import time
import tqdm
import argparse

# Torch
from torch.utils.data import DataLoader
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn

# Sci-Kit
from sklearn.model_selection import train_test_split

# Custom Imports
from data import DeviceDataset
from networks import *

class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        # Paramaters
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = self.get_device()
        
        # Loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Test Values
        self.test_loss = None
        self.test_accuracy = None
        
        # First
        self.first = True
    
    def set_loaders(self, train_loader, val_loader, test_loader):
        """
        Set the data loaders for training, validation, and testing.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

        Returns:
            None
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train(self, epochs, validate=True, validate_iter=1, test=True):
        """
        Train the model using the provided training data loader and validate using the validation data loader.
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            epochs (int): Number of epochs to train the model.
            validate (bool, optional): Whether to perform validation after each epoch. Default is True.
            validate_iter (int, optional): Frequency of validation in terms of epochs. Default is 1.
        Returns:
            None
        """
        self.move_to_device(self.device)
        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

            if validate:
                if (epoch+1) % validate_iter == 0:
                    val_loss, val_accuracy = self.validate(val_loader)
                    self.printer(epoch+1, epochs, val_loss, val_accuracy, self.first)
                    if self.first: self.first = False
        
        # Test the Model
        if test:
            test_loss, test_accuracy = self.test(test_loader)
            print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        
        # Move Model Back to CPU to Conserve Memory
        self.move_to_device(torch.device("cpu"))
        self.first = True

    def save_model(self, path, name):
        """
        Save the model to the specified path.

        Args:
            path (str): Path to save the model.

        Returns:
            None
        """
        # Full Path, Add Suffix.
        path = path + name + ".pt"
        
        torch.save(self.model.state_dict(), path)

    def validate(self, val_loader):
        """
        Validate the model on the validation dataset.

        Args:
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

        Returns:
            tuple: A tuple containing:
                - val_loss (float): The average loss on the validation dataset.
                - val_accuracy (float): The accuracy on the validation dataset.
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                val_loss += self.loss_fn(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / len(val_loader.dataset)
        return val_loss, val_accuracy

    def test(self, test_loader):
        """
        Test the model on the test dataset.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

        Returns:
            tuple: A tuple containing:
                - test_loss (float): The average loss on the test dataset.
                - test_accuracy (float): The accuracy on the test dataset.
        """
        print(f'--------------------------------------------')
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.loss_fn(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = correct / len(test_loader.dataset)
        
        self.test_loss = f'{test_loss:.3f}'
        self.test_accuracy = f'{test_accuracy:.3f}'
        
        return self.test_loss, self.test_accuracy
    
    def get_device(self):
        """
        Determines the appropriate device for PyTorch operations.

        This method checks for the availability of CUDA and XPU devices in that order.
        If neither is available, it defaults to the CPU.

        Returns:
            torch.device: The device to be used for PyTorch operations.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.xpu.is_available():
            return torch.device("xpu")
        else:
            return torch.device("cpu")

    def move_to_device(self, device):
        self.model.to(device)

    def printer(self, epoch, total_epochs, loss, accuracy, first):
        if first: print(f'____________________________________________')
        if first: print(f"|    EPOCH     |    LOSS    |   ACCURACY   |")
        # Get Number of Characters For Each Item Except first.
        epoch_len = len(f'{epoch}')
        total_epochs_len = len(f'{total_epochs}')
        loss = f'{loss:.4f}'
        accuracy = f'{accuracy:.4f}'
        
        # Add x Number of 0s to make epoch 4 characters long.
        # For Example, if Epoch is 1, make it "0001". If it is 154, make it "0154"
        epoch = str(epoch).zfill(4)
        total_epochs = str(total_epochs).zfill(4)
        epoch_char = f"{epoch}/{total_epochs}"
        
        print(f'|  {epoch_char}   |   {loss}   |    {accuracy}    |')
    
class MNIST_Dataloader:
    def __init__(self, opt, dir="data/", batch_size=32, train_split=0.6, val_split=0.2):
        # Basic Parameters
        self.dir = dir
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        
        # Load Data
        self.images = None
        self.labels = None
        self.load_data()
        
        # Prep The Images
        self.prep_images()
        
        # Split the Data According to train_split and val_split... test_split is 1 - train_split - val_split
        # Temporary Variables for Splitting Data
        train_images, test_images, train_labels, test_labels = train_test_split(self.images, self.labels, test_size=1-self.train_split)
        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=self.val_split/(self.train_split + self.val_split))
        
        # Load This Data into Datasets/Dataloaders
        self.train_dataset = DeviceDataset(train_images, train_labels, torch.device("cpu"))
        self.test_dataset = DeviceDataset(test_images, test_labels, torch.device("cpu"))
        self.val_dataset = DeviceDataset(val_images, val_labels, torch.device("cpu"))
    
    def send_datasets_to_device(self, device):
        self.train_dataset.to_device(device)
        self.test_dataset.to_device(device)
        self.val_dataset.to_device(device)
    
    def get_dataloaders(self, batch_size=32, shuffle=True):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader, val_loader, test_loader
    
    def load_data(self):
        data = MNIST(self.dir)
        self.images, self.labels = data.load_training()
    
    def prep_images(self):
        self.images = np.array(self.images)
        self.images = self.images.reshape(-1, 28, 28).astype(np.float32)
        self.images = self.images / 255.0
        self.images[self.images > 0] = 1.0
        self.images = np.expand_dims(self.images, axis=1)

def check_args(opt):
    # Make sure that opt.train_split + opt.val_split is not greater than 1
    if opt.train_split + opt.val_split > 1:
        raise ValueError("'--train_split' + '--val_split' must be less than or equal to 1")

def arg_parse():
    parser = argparse.ArgumentParser()
    
    # Add --epochs
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model.")
    
    # Add --batch_size
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    
    # Add --learning_rate
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    
    # Add --val_split
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to use for validation.")
    
    # Add --data_dir
    parser.add_argument("--data_dir", type=str, default="data/", help="Directory containing the MNIST dataset.")
    
    # Add --train_split
    parser.add_argument("--train_split", type=float, default=0.6, help="Fraction of data to use for training.")
    
    # Add --shuffle (Boolean)
    parser.add_argument("--shuffle", type=bool, default=True, help="Whether to shuffle the data before splitting.")
    
    # Add --network_name
    parser.add_argument("--network_name", type=str, default="SimpleModel", help="Name of the network to use.")
    
    # Add --do_validate (Boolean)
    parser.add_argument("--do_validate", type=bool, default=True, help="Whether to validate the model.")
    
    # Add --do_test (Boolean)
    parser.add_argument("--do_test", type=bool, default=True, help="Whether to test the model.")
    
    # Add --validate_iter
    parser.add_argument("--validate_iter", type=int, default=1, help="Frequency of validation in terms of epochs.")
    
    # Add --save_model (Boolean)
    parser.add_argument("--save_model", type=bool, default=False, help="Whether to save the model.")
    
    # Add --model_path
    parser.add_argument("--model_path", type=str, default="models/", help="Path to save the model.")

    return parser.parse_args()

if __name__ == "__main__":
    # Get the Command Line Arguments
    opt = arg_parse()
    check_args(opt)
    
    # Print out the Arguments Individually
    print(f'#############################################')
    print(f"## ARGUMENTS ##")
    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")
    print(f'#############################################\n')
    
    # Get Model, Optimizer and Loss Function
    wrapper = NetworkWrapper(opt.network_name, print_info=True)
    model = wrapper.model
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    # Get Data
    data = MNIST_Dataloader(opt, batch_size=opt.batch_size, train_split=opt.train_split, val_split=opt.val_split)
    
    # Create an Instance of Trainer
    trainer = Trainer(model, optimizer, loss_fn)
    
    # Send the Datasets to the Device
    data.send_datasets_to_device(trainer.device)
    
    # Dataloaders
    train_loader, val_loader, test_loader = data.get_dataloaders(batch_size=opt.batch_size, shuffle=opt.shuffle)
    
    # Set the Data Loaders
    trainer.set_loaders(train_loader, val_loader, test_loader)
    
    # Train the Model
    trainer.train(opt.epochs, validate=opt.do_validate, validate_iter=opt.validate_iter, test=opt.do_test)
    
    # Save the Model
    if opt.save_model:
        # Check to see if opt.do_test was true.
        if opt.do_test:
            accuracy = f'{trainer.test_accuracy}'
            date = ""
        else:
            accuracy = ""
            date = time.strftime("%Y%m%d_%H%M%S")
        
        name = opt.network_name + "_" + accuracy + date
        trainer.save_model(opt.model_path, name)