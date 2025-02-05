import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class DeviceDataset(Dataset):
    def __init__(self, image_list, label_list, device):
        self.device = device
        self.data = []
        self.labels = []

        # Load all images into memory and transfer to the specified device
        for img in image_list:
            self.data.append(torch.tensor(img).to(self.device))

        # Transfer labels to the specified device
        self.labels = torch.tensor(label_list).to(self.device)
    
    def to_device(self, device):
        self.device = device
        self.data = [img.to(self.device) for img in self.data]
        self.labels = self.labels.to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]