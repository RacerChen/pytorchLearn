# How to adjust learning rate to realize higher model performance
"""
torch.optim.lr_scheduler: provides several methods to adjust the learning rate based on the number of epochs
torch.optim.lr_scheduler.ReduceLROnPlateau allows dynamic learning rate reducing based on some validation measurements
Pytorch documentation: https://pytorch.org/docs/stable/optim.html
"""

import torch.nn as nn
import torch
import torch.optim.lr_scheduler as lr_schduler

# Cautions: learning rate scheduling should be applied [after optimizer's update], code structures should like this:
'''
scheduler = ...
for epoch in range(100):
    train(xxx)
    validate(xxx)
    scheduler.step()
'''

# demo1: lr_schduler.LambdaLR
lr = 0.1
model = nn.Linear(10, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Calculate the lr by lambda functions
lambda1 = lambda epoch: epoch / 10  # the higher the epoch is, the higher the lr is.

schduler = lr_schduler.LambdaLR(optimizer, lr_lambda=lambda1)

print(optimizer.state_dict())
for epoch in range(5):
    optimizer.step()
    schduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])

'''
output:
{'state': {}, 'param_groups': [{'lr': 0.0, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'initial_lr': 0.1, 'params': [0, 1]}]}
0.010000000000000002
0.020000000000000004
0.03
0.04000000000000001
0.05
'''


# demo2: lr_schduler.MultiplicativeLR
lr2 = 0.1
model2 = nn.Linear(10, 1)

optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr2)

# Calculate the lr by multiply
lambda2 = lambda epoch: 0.95  # the higher the epoch is, the higher the lr is.

schduler2 = lr_schduler.MultiplicativeLR(optimizer2, lr_lambda=lambda2)

print(optimizer2.state_dict())
for epoch in range(5):
    optimizer2.step()
    schduler2.step()
    print(optimizer2.state_dict()['param_groups'][0]['lr'])
'''
output:
{'state': {}, 'param_groups': [{'lr': 0.1, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'initial_lr': 0.1, 'params': [0, 1]}]}
0.095
0.09025
0.0857375
0.08145062499999998
0.07737809374999999
'''

# demo3: lr_schduler.StepLR
# Decays the learning rate of each paramter group by gamma every step_size epochs.
lr3 = 0.1
model3 = nn.Linear(10, 1)

optimizer3 = torch.optim.Adam(model3.parameters(), lr=lr3)

# Calculate the lr by multiply
lambda3 = lambda epoch: 0.95  # the higher the epoch is, the higher the lr is.

schduler3 = lr_schduler.StepLR(optimizer3, step_size=30, gamma=0.1)

print(optimizer3.state_dict())
for epoch in range(90):
    optimizer3.step()
    schduler3.step()
    print(optimizer3.state_dict()['param_groups'][0]['lr'])
'''
output:
lr = 0.5, epoch < 30
lr = 0.05, 30<= epoch < 60
lr = 0.005, 60 <= epoch < 90
'''

# demo4: MultiStepLR
# scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0,1)
'''
output:
lr = 0.5, epoch < 30
lr = 0.05, 30<= epoch < 80
lr = 0.005, 80 <= epoch < 90
'''

# demo5: CosineAnnealingLR

# demo6: ReduceLROnPlateau
# Reduce learning rate when a metric has stopped improving.
# Model often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
# This schduler reads a metrics quantity and if no improvement is seen for a 'patience' number of epochs,
# the learning rate is reduces.
'''
torch.optim.lr_schduler.ReduceLROnPalteau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001)
[mode]: One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing;
 in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
[factor]: Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
[patience (int)]: Number of epochs with no improvement after which learning rate will be reduced.
 For example, if patience = 2, then we will ignore the first 2 epochs with no improvement,
  and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
'''