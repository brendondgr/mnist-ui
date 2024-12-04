from torch.utils.data import Dataset
from mlxtend.data import loadlocal_mnist

class CustomDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data, self.labels = self.load_data(data_path, label_path)
        
        # Convert to Torch 
        self.data, self.labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def load_data(self, data_path, label_path):
        return loadlocal_mnist(images_path=data_path, labels_path=label_path)