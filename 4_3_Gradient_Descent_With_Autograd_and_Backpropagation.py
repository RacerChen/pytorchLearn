# 3 Using Pytorch to do Gradient Descent
"""
Outline:
                            1       2           3               4
Prediction:             |Manual|Manual  |Manual           |PyTorch Model    |
Gradients Computation:  |Manual|Autograd|Autograd         |Autograd         |
Loss Computation:       |Manual|Manual  |PyTorch Loss     |PyTorch Loss     |
Parameter Updates:      |Manual|Manual  |PyTorch Optimizer|PyTorch Optimizer|
"""

import torch
import torch.nn as nn
'''
step
1) Design model (input, output size, forward pass)
2) Construct loss and optimizer
3) Training loop
    - forward pass: compute prediction
    - backward pass: gradients
    - update weights
'''
# f = w * x
# f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)  # initialize weight


# model prediction
def forward(x):
    return w * x


print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)


for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gredients = backward pass
    l.backward()  # dl/dw

    # update weights
    optimizer.step()

    # zero gradient
    optimizer.zero_grad()

    if epoch % 5 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
