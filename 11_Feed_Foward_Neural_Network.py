# Use pytorch to do a whole ML project

# MNIST
# Dataloader, Transformation
# Multilayer Neural Net, activation function
# loss and optimizer
# Training Loop(Batch training)
# Model evaluation
# GPU support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Solution for OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')

# hyper parameters
input_size = 784  # 28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.01

# MNIST dataset
training_dataset = torchvision.datasets.MNIST(root='./Data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./Data', train=True, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=False)

example = iter(train_loader)
samples, labels = next(example)
print(samples.shape, labels.shape)

# print dataset
for i in range(6):
    plt.subplot(2, 3, i + 1)  # 2 rows, 3 columns， 在第i+1块绘图
    plt.imshow(samples[i][0], cmap='gray')


# plt.show()


class NeuralNet(nn.Module):
    def __init__(self, input_layer, hidden_layer, _num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_layer, hidden_layer)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer, _num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # original images shape 100, 1, 28, 28
        # network needs shape 100, 728
        images = images.reshape(-1, input_size).to(device)  # if gpu is working, push to it
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loos={loss:.5f}')


# test

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in train_loader:
        # original images shape 100, 1, 28, 28
        # network needs shape 100, 728
        images = images.reshape(-1, input_size).to(device)  # if gpu is working, push to it
        labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, dim=1)  # return the max value's index, dim=1 返回每行的最大值，0为每列的最大值
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

acc = 100.0 * n_correct / n_samples
print(f'acc={acc:.4f}%')
