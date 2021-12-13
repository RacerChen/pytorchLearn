# learning backpropagation in pytorch
import torch

# Chain Rule
'''
x->a(x)->y->b(y)->z
dz/dx = dz/dy * dy/dx
'''

# Computational Graph
'''
x
 \
 x*y->z
 /
y
dz/dx = dxy/dx = y
dz/dy = dxy/dy = x
'''

# with chain rule
# Finally, dloss/dx

# Steps of machine learning
'''
1) Forward pass: computing loss
2) Compute local gradients
3) Backward pass: compute dLoss/dWeights using the chain rule
'''

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y)**2

print(loss)

# backward pass
loss.backward()
print(w.grad)

# Output:
# tensor(1., grad_fn=<PowBackward0>)
# tensor(-2.)

# backpass, changing weight
# weight_new = weight_old - grad

# next round train
