import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class water2d(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = torch.from_numpy(np.load(data_path)).permute(0, 1, 4, 2, 3)
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
    train_dataset = water2d(data_path=data_root + 'train_100_water2d.npy', transform=None)
    test_dataset = water2d(data_path=data_root + 'test_100_water2d.npy', transform=None)
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_workers)
    dataloader_validation = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                       num_workers=num_workers)
    dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std

if __name__ == "__main__":
    dataloader_train, dataloader_validation, dataloader_test, mean, std = load_data(batch_size=20, 
                                                                                    val_batch_size=20, 
                                                                                    data_root='/data/workspace/yancheng/MM/neural_manifold_operator/data/2DSWE/', 
                                                                                    num_workers=8)
    for input_frames, output_frames in dataloader_train:
        print(input_frames.shape, output_frames.shape)


