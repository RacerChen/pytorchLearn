# Leaning Softmax and Cross Entropy in Pytorch
import torch
import numpy as np
import torch.nn as nn

"""
Softmax:
# Generate 1-scaled probabilities of score, the the max one's index is the Y_pred
Usually, use e exponent
Linear Layer     Scores/Logits       Softmax       Probabilities
----------->        2.0             ----------->        0.66
----------->        1.0             ----------->        0.24         -----------> Y_pred
----------->        0.1             ----------->        0.10
"""


# code with numpy
def softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis=0)


x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpu:', outputs)

# pytorch version
x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
# dim means the dimension which sum equals 1
# detail explanation: https://blog.csdn.net/sunyueqinghit/article/details/101113251
print('softmax numpu:', outputs)


"""
Cross-Entropy
D(Y', Y) = -1/N*sum(yi,log(y'i))
Y' is the output of softmax function

One-hot tag: y = [1,0,0]
Y'1 = [0.7, 0.2, 0.1]
Y'2 = [0.1, 0.3, 0.6]

D(Y'1, Y) = 0.35
D(Y'2, Y) = 2.30
The later one is worse
"""


# code with numpy
def cross_entropy(predicted, actual):
    return -np.sum(actual * np.log(predicted))


Y = np.array([1, 0, 0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
print(f'good cross entropy: {cross_entropy(Y_pred_good, Y):.5f}')
print(f'bad cross entropy: {cross_entropy(Y_pred_bad, Y):.5f}')
# Console:
# good cross entropy: 0.35667
# bad cross entropy: 2.30259

# pytorch version
loss = nn.CrossEntropyLoss()
# Cautions:
# 1)
# nn.CrossEntropyLoss() already applied nn.LogSoftmax and nn.NLLoss(negative log likelihood loss)
# so, No Softmax in last layer anymore
# 2)
# Y has class labels, not ONE-HOT
# Y_pred use raw score(logits), no Softmax

Y_tag = torch.tensor([0])  # no softmax, class labels directly
# Our example, nsamples x nclasses = 1 x 3
Y_pred_good_raw = torch.tensor([[2.0, 1.1, 0.6]])  # logits directly, not softmax
Y_pred_bad_raw = torch.tensor([[0.5, 2.1, 0.3]])  # logits directly, not softmax
l1 = loss(Y_pred_good_raw, Y_tag)
l2 = loss(Y_pred_bad_raw, Y_tag)
print(f'nn good cross entropy: {l1.item():.5f}')
print(f'nn bad cross entropy: {l2.item():.5f}')
# Console:
# nn good cross entropy: 0.50269
# nn bad cross entropy: 1.91276

_, predictions_good = torch.max(Y_pred_good_raw, 1)
_, predictions_bad = torch.max(Y_pred_bad_raw, 1)
print(predictions_good)
print(predictions_bad)
# Console:
# tensor([0])
# tensor([1])

# loss of pytorch supports multiple samples
# Our example now, nsamples x nclasses = 3 x 3
Ys = torch.tensor([2, 0, 1])
Ys_pred_good = torch.tensor([[0.05, 0.15, 0.8], [0.7, 0.2, 0.1], [0.3, 0.6, 0.1]])
Ys_pred_bad = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.8, 0.1, 0.1]])

l1 = loss(Ys_pred_good, Ys)
l2 = loss(Ys_pred_bad, Ys)
print(l1.item())
print(l2.item())
# Console:
# 0.7705284953117371
# 1.3703209161758423


# Multiclass problem
class NerualNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NerualNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(input_size, num_classes)  # no softmax if we use nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


model = NerualNet(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()  # contains softmax already


# But if it is a binary problem, eg. Is it a dog
# in pyTorch we use nn.BCELoss()  // binary cross entropy loss
# but now we use a sigmoid function at last
class NerualNetBinary(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NerualNetBinary, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(input_size, 1)  # binary output just need one dimension

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        y_pred = torch.sigmoid(out)
        return y_pred


modelBinary = NerualNetBinary(input_size=28*28, hidden_size=5)
criterionBinary = nn.BCELoss()
