# Learning Activation Function in pyTorch
import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Activation function apply a non-linear transformation and decide whether a neuron should be activate or not.
Reason for use activation function: without activation functions, our network is basically just a stacked linear
regression model, whose performance degraded sharply when faced with difficult tasks.

Most popular activation functions:
1) step function:
                1, if x >= 0
     f(x) = {
                0, otherwise
    |
  1 |         __________
  0 |_________|
    |
    |____________________
    -         0         +
2) Sigmoid:
    f(x) = 1 / (1+e^-x)
    |
  1 |         ___------
  0 |______---
    |
    |____________________
    -         0         +
    range from 0~1, typically in the last layer of a binary classification problem
3) Tanh:
    f(x) = 2/(1+e^-2x) - 1
    |
  1 |          ___------
  0 |        __
 -1 |_____---
    |
    |____________________
    -         0         +
    Range from -1~-1,Basically a scaled sigmoid function, usually use for hidden layers
4) ReLU
    f(x) = max(0, x)
  2 |             /
  1 |           /
  0 |_________/
    |
    |
    |____________________
    -         0 1 2     +
    linear when x > 0, if we don't know what to use, just use a ReLU for hidden layers
5) Leaky ReLU
              x, if x >= 0
     f(x) = {
              ax, otherwise
  2 |             /
  1 |           /
  0 |         /
 -1 |_____---'
    |
    |____________________
    -         0 1 2     +
    usually, a is small, like 0.01, tries to solve the vaninshing gradient problem.
    In ReLU, if logits is negative, no grad is generate, so the weight cannot be refreshed.
6) Softmax
"""


# 2 ways to use activation function
class LogisticRegression1(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression1, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out


class LogisticRegression2(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression2, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        out = self.linear(x)
        out = torch.sigmoid(out)
        # or
        # out = F.sigmoid(out)
        # some activation function can only be found in F, like leaky_ReLU
        return out

