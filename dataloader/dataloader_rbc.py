import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as data_utils


class rbc_data(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = torch.load(data_path)
        self.transform = transform
        self.mean = 0
        self.std = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_frames = self.data[idx][:10]
        output_frames = self.data[idx][10:]
        return input_frames, output_frames


def load_data(batch_size, val_batch_size, data_root, num_workers):
    dataset = rbc_data(data_path=data_root + 'rbc_data_combined.pt', transform=None)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = data_utils.random_split(dataset, [train_size, val_size, test_size])
    
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_workers)
    dataloader_validation = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                       num_workers=num_workers)
    dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std


if __name__ == "__main__":
    data_path = '/data/workspace/yancheng/MM/neural_manifold_operator/data/'
    dataloader_train, dataloader_validation, dataloader_test, mean, std = load_data(batch_size=1, 
                                                                                val_batch_size=1, 
                                                                                data_root=data_path, 
                                                                                num_workers=8)
    for input_frames, output_frames in dataloader_train:
        print(input_frames.shape, output_frames.shape)