import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class APDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        data = scipy.io.loadmat(path)
        self.W = data['Wsav'] # shape: (N, X, Y, T)
        self.V = data['Vsav'] # shape: (N, X, Y, T)
        self.B = data['B'][:, 0] # shape: (N,)
        self.t = data['tend'].flatten() # shape: (1,)
        self.x = data['geom'].flatten()[:-2] # shape: (X,)
        self.t = np.arange(1, self.t + 1) # shape: (T,)
        X, Y, T = np.meshgrid(self.x, self.x, self.t) # shapes: (X, Y, T)
        X = np.expand_dims(X, axis=0)  # Shape: (1, X, Y, T)
        Y = np.expand_dims(Y, axis=0)  # Shape: (1, X, Y, T)
        T = np.expand_dims(T, axis=0)  # Shape: (1, X, Y, T)
        X = np.repeat(X, self.W.shape[0], axis=0)  # Shape: (N, X, Y, T)
        Y = np.repeat(Y, self.W.shape[0], axis=0)  # Shape: (N, X, Y, T)
        T = np.repeat(T, self.W.shape[0], axis=0)  # Shape: (N, X, Y, T)
        self.data = np.stack((self.V, self.W, X, Y, T), axis=-1) # Shape: (N, X, Y, T, 5)

    def __len__(self):
        return self.W.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.B[idx]

def get_dataloaders(path):
    # Load the dataset
    dataset = APDataset(path)

    # Split the dataset into training and testing sets (80/20 split)
    train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)

    # Create subsets for training and testing
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
