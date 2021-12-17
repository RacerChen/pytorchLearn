# Learning how to use TensorBoard for visualization
"""
TensorBoard is developed by TensorFlow guys, but can be used in pytorch also.

Functions:
1) Tracking and visualizing metrics such as loss and accuracy
2) Visualizing the model graph(ops and layers)
3) Viewing histograms of weights, biases, or other tensors as they change over time
4) Projecting embeddings to a lower dimensional space
5) Displaying images, text, and audio data
6) Profiling TensorFlow programs
7) So on

Next we will use 11_Feed_Forward_Neural_Network.py to util TensorFlow

How to activate TensorBoard?
'tensorboard --logdir=runs' in Terminal

"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import sys
writer = SummaryWriter('runs/mnist_pr')
# a writer writes a chart, can been seen in different colors

# Solution for OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')

# hyper parameters
input_size = 784  # 28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset
training_dataset = torchvision.datasets.MNIST(root='./Data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./Data', train=True, transform=transforms.ToTensor(), download=False)

train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=False)

example = iter(train_loader)
samples, labels = next(example)
samples.to(device)
print(samples.shape, labels.shape)

# print dataset in tensorboard
# for i in range(6):
#     plt.subplot(2, 3, i + 1)  # 2 rows, 3 columns， 在第i+1块绘图
#     plt.imshow(samples[i][0], cmap='gray')
# img_grid = torchvision.utils.make_grid(samples)  # create a image showing grid with images to show
# writer.add_image('mnist_images', img_grid)  # writing in the board


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

writer.add_graph(model, samples.reshape(-1, input_size))


n_total_steps = len(train_loader)

a = []
b = []
c = []
running_loss = 0.0
running_correct = 0

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

        running_loss += loss.item()

        _, predictions = torch.max(outputs, dim=1)
        running_correct += (predictions == labels).sum().item()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss={loss:.5f}')
            writer.add_scalar('traning loss', running_loss / 100, epoch * n_total_steps + i)  # scalar colunm
            # running_loss divide 100 cause we print in every 100 steps,
            # and epoch * n_total_steps + 1 is the absolute step
            writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i)
            a.append(running_loss)
            b.append(running_correct)
            c.append(epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0

print(a)
print(b)
print(c)
# test

labels_list = []
preds_list = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        # original images shape 100, 1, 28, 28
        # network needs shape 100, 728

        images = images.reshape(-1, input_size).to(device)  # if gpu is working, push to it
        labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, dim=1)  # return the max value's index, dim=1 返回每行的最大值，0为每列的最大值
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        class_prediction = [F.softmax(output, dim=0) for output in outputs]

        preds_list.append(class_prediction)
        labels_list.append(labels)
    preds = torch.cat([torch.stack(batch) for batch in preds_list])
    labels = torch.cat(labels_list)

acc = 100.0 * n_correct / n_samples
print(f'acc={acc:.4f}%')

print(preds)
print(labels)

classes = range(10)
for i in classes:
    label_i = labels == i
    preds_i = preds[:, i]  # colums whose index == i, the probability value of predict number i
    writer.add_pr_curve(str(i), label_i, preds_i, global_step=0)  # curve of precision and recall in different threshold

writer.close()