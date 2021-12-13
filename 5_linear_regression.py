# Linear Regression leaning in pytorch
"""
step
1) Design model (input, output size, forward pass)
2) Construct loss and optimizer
3) Training loop
    - forward pass: compute prediction
    - backward pass: gradients
    - update weights
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape


# 1) model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) loss and optimizer
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass
    y_pred = model(X)

    # calculate loss
    loss = criterion(y_pred, y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    # zero grad
    optimizer.zero_grad()

    if epoch % 5 == 0:
        print(f'epoch:{epoch}, loss:{loss.item():.4f}')

# plot
predicted = model(X).detach().numpy()  # use detach, because we don't want calculate grad here
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.title('Linear Regression')
plt.show()
