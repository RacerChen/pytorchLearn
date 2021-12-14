# Learning how to load dataset using dataloader
"""
eg
data = numpy.loadtxt('wine.csv)

# traning loop
for epoch in range(steps):
    x, y = data

this load the whole dataset one time,
which cause to much time to calculate grad
and sometimes memory is limited, cannot load big dataset
so we use Dataset and DataLoader to load dataset and train/test with batch
"""

"""
Parameters:
epoch = 1 forward and backward pass if All training samples
batch_size = number of traning samples in one [iteration]
number of iteration = number of passes, each pass using [batch_size] number of samples

eg.100 samples, batch_size=20, then 5 iteration a epoch
"""

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])  # get all row by col >= 1
        self.y = torch.from_numpy(xy[:, [0]])  # get all row by col 0
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[index]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


if __name__ == '__main__':

    dataset = WineDataset()

    first_data = dataset[0]

    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)
    # shuffle make the data in batch random, number workers make the loading process faster using multiple threads
    # Cautions: if num_workers > 0, the dataloader must written in the main function

    # return a batch a time
    # dataiter = iter(dataloader)
    # data = dataiter.next()
    # features, labels = data
    # print(features.shape[0])
    # print(features, labels)


