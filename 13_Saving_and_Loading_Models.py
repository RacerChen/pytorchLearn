import torch
import torch.nn as nn
"""
1
Basic functions:
torch.save(arg, PATH)  # save any dictionary, data is serial, not human readabel
torch.load(PATH)
model.load_state_dict(arg)

2
Save model:
Complete model
torch.save(model, PATH)

model = torch.load(PATH)  # model must be define somewhere
model.eval()

3
State Dict
torch.save(model.state_dict(), PATH)

# model must be created again with parameters
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
"""


class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = Model(n_input_features=6)

# [Lazy method]
FILE = "model_state_dict.pth"
# torch.save(model, FILE)
# model = torch.load(FILE)
# model.eval()
# for param in model.parameters():
#     print(param)


# [Preferred way]
# torch.save(model.state_dict(), FILE)
# loaded_model = Model(n_input_features=6)
# loaded_model.load_state_dict(torch.load(FILE))
# for param in model.parameters():
#     print(param)

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# print(model.state_dict())
# print(optimizer.state_dict())

checkpoint = {
    'epoch': 90,
    'model_state': model.state_dict(),
    'optim_state': optimizer.state_dict()
}

# torch.save(checkpoint, 'checkpoint.pth')

loaded_checkpoint = torch.load('checkpoint.pth')
epoch = loaded_checkpoint['epoch']

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])

print(optimizer.state_dict())


'''
# [Save on GPU, load on CPU]
device = torch.device('cuda')
model.to(device)
torch.save(model.state_dict(), PATH)

device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
--------------------------------------------------------------
# [Save on GPU, load on GPU]
device = torch.device('cuda')
model.to(device)
torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
model.to(device)
---------------------------------------------------------------
# [Save on CPU, load on GPU]
torch.save(model.state_dict(), PATH)

model.load_state_dict(torch.load(PATH, map_location='cuda:0'))  # load the 0th gpu
device = torch.device('cuda')
model = Model(*args, **kwargs)
model.to(device)
'''