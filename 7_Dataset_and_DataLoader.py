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

    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples/4)
    print(total_samples, n_iterations)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            if i % 5 == 0:
                print(f'epoch: {epoch+1}/{num_epochs}, step{i+1}/{n_iterations}, input {inputs.shape}')



'''
Console:
178 45
epoch: 1/2, step1/45, input torch.Size([4, 13])
epoch: 1/2, step6/45, input torch.Size([4, 13])
epoch: 1/2, step11/45, input torch.Size([4, 13])
epoch: 1/2, step16/45, input torch.Size([4, 13])
epoch: 1/2, step21/45, input torch.Size([4, 13])
epoch: 1/2, step26/45, input torch.Size([4, 13])
epoch: 1/2, step31/45, input torch.Size([4, 13])
epoch: 1/2, step36/45, input torch.Size([4, 13])
epoch: 1/2, step41/45, input torch.Size([4, 13])
epoch: 2/2, step1/45, input torch.Size([4, 13])
epoch: 2/2, step6/45, input torch.Size([4, 13])
epoch: 2/2, step11/45, input torch.Size([4, 13])
epoch: 2/2, step16/45, input torch.Size([4, 13])
epoch: 2/2, step21/45, input torch.Size([4, 13])
epoch: 2/2, step26/45, input torch.Size([4, 13])
epoch: 2/2, step31/45, input torch.Size([4, 13])
epoch: 2/2, step36/45, input torch.Size([4, 13])
epoch: 2/2, step41/45, input torch.Size([4, 13])

notes:
2 epochs
4 batch size
178 total samples
so
ceil(178/4) = 45 iteration per epoch
'''

