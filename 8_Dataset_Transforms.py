# Dataset Transform in Pytorch
# Transform is used for transform other formats of data into tensor like txt, image, and so on
# Official document is in: https://pytorch.org/docs/master/generated/torch.nn.Transformer.html#transformer


"""
On images
---------
CenterCrop, Grayscale, Pad, RandomAffine,
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
__________
ToPILImage: from tensor or ndrarray
ToTensor: from numpy.ndarray or PILImage

Generic
_______
Use Lambda

Custom
______
Write own class

Compose multiple Transforms
___________________________
eg. composed = transforms.Compose([Rescale(256), RandomCrop(224)])

torchvision.transforms.ReScale(256)
torchvision.transforms.ToTensor()
"""
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)

        # transform use tools
        # self.x = torch.from_numpy(xy[:, 1:])  # get all row by col >= 1
        # self.y = torch.from_numpy(xy[:, [0]])  # get all row by col 0

        # no transform here, leave it as np array
        self.x = xy[:, 1:]  # get all row by col >= 1
        self.y = xy[:, [0]]  # get all row by col 0

        self.n_samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        # dataset[index]
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, samples):
        inputs, target = samples
        inputs *= self.factor
        return inputs, target


dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
# Console: <class 'torch.Tensor'> <class 'torch.Tensor'>


composed = torchvision.transforms.Compose([ToTensor(), MulTransform(1.5)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

