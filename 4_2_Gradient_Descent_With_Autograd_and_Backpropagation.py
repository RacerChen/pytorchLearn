# 2 Using Pytorch to do Gradient Descent
"""
Outline:
                            1       2           3               4
Prediction:             |Manual|Manual  |Manual           |PyTorch Model    |
Gradients Computation:  |Manual|Autograd|Autograd         |Autograd         |
Loss Computation:       |Manual|Manual  |PyTorch Loss     |PyTorch Loss     |
Parameter Updates:      |Manual|Manual  |PyTorch Optimizer|PyTorch Optimizer|
"""

import torch

# f = w * x

# f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)  # initialize weight


# model prediction
def forward(x):
    return w * x


# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gredients = backward pass
    l.backward()  # dl/dw
    dw = w.grad

    # update weights manually (can be replaced by SGD)
    with torch.no_grad():
        w -= learning_rate * dw

    # zero gradient
    w.grad.zero_()

    if epoch % 5 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
# Comparing to np, autograd need more epochs to reach the same accuracy, so the calculation precision is less
