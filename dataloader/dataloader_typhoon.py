import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from numpy.lib.stride_tricks import as_strided
import torch.nn.functional as F


class typhoon(Dataset):
    def __init__(self, data_path, transform=None, subset_size=10000):
        self.data = np.load(data_path)
        self.transform = transform
        self.window_size = 50
        self.step_size = 1
        self.num_channels = 3
        self.height = 512
        self.width = 512
        self.num_samples = (self.data.shape[0] - self.window_size) // self.step_size + 1
        self.strides = self.data.strides
        self.new_shape = (self.num_samples, self.window_size, self.num_channels, self.height, self.width)
        self.new_strides = (self.step_size * self.strides[0],) + self.strides  
        self.sliding_window_data = as_strided(self.data, shape=self.new_shape, strides=self.new_strides)  


        subset = np.random.choice(self.data.flatten(), subset_size)
        self.mean = np.mean(subset)
        self.std = np.std(subset)


    def __len__(self):
        return len(self.sliding_window_data)

    def __getitem__(self, idx):
        input_frames = self.sliding_window_data[idx][:10]
        output_frames = self.sliding_window_data[idx][10:20]

        # Convert the numpy arrays to torch tensors
        input_frames = torch.tensor(input_frames, dtype=torch.float)
        output_frames = torch.tensor(output_frames, dtype=torch.float)

        # Rescale the frames to 256x256
        input_frames = F.interpolate(input_frames, size=(128, 128))
        output_frames = F.interpolate(output_frames, size=(128, 128))

        # Normalize the frames
        input_frames = (input_frames - self.mean) / self.std
        output_frames = (output_frames - self.mean) / self.std

        return input_frames, output_frames



# data_root = "/data/workspace/yancheng/MM/AAAA/Processing_Data/"
def load_data(batch_size, val_batch_size, data_root, num_workers):
    train_dataset = typhoon(data_path=data_root + "resized_data_A_512.npy", transform=None)
    val_dataset = typhoon(data_path=data_root + "resized_data_U_512.npy", transform=None)
    test_dataset = typhoon(data_path=data_root + "resized_data_U_512.npy", transform=None)
    

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_workers)
    dataloader_validation = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                       num_workers=num_workers)
    dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std


if __name__ == "__main__":
    data_path = "/data/workspace/yancheng/MM/AAAA/Processing_Data/"
    dataloader_train, dataloader_validation, dataloader_test, mean, std = load_data(batch_size=20, 
                                                                                val_batch_size=20, 
                                                                                data_root=data_path, 
                                                                                num_workers=8)
    # for input_frames, output_frames in dataloader_validation:
    #     print(input_frames.shape, output_frames.shape)
    

    print(len(dataloader_train))