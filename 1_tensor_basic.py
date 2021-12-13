import torch
import numpy as np

# 【Part I: tensor initialize】
# empty tensor: not initialized
# 1-D
x_empty_1d = torch.empty(2)
print(x_empty_1d)

# 2-D
x_empty_2d = torch.empty(2, 2)
print(x_empty_1d)


# random initialized tensor
x_rand_1d = torch.rand(3)
print(x_rand_1d)
x_rand_2d = torch.rand(2, 2)
print(x_rand_2d)

# zero initialized tensor
x_zero_2d = torch.zeros(2, 2)
print(x_zero_2d)


# one initialized tensor
x_one_2d = torch.ones(2, 2)
print(x_one_2d)


# get torch dtype, default
print(x_one_2d.dtype)


# assign dtype to a tensor
x_one_2d_int = torch.ones(2, 2, dtype=torch.int)
print(x_one_2d_int.dtype)


# get tensor size
print(x_one_2d.size())


# initialized tensor from list
x_tensor_list = torch.tensor([1.2, 3.2, 4.5])
print(x_tensor_list)


# 【Part II: tensor calculation】
# adding directly
x_add = x_one_2d + x_zero_2d
print(x_add)
# adding function
x_add_ = torch.add(x_one_2d, x_zero_2d)
print(x_add_)
# adding inplace
x_rand_2d.add_(x_one_2d)
print(x_rand_2d)

# minus
# z = x - y # z = torch.sub(x, y) # x.sub_(y)

# mul
# z = x * y # z = torch.mul(x, y) # x.mul_(y)

# div
# z = x / y # z = torch.mul(x, y) # x.mul_(y)


# 【Part III: Part slice】
x = torch.rand(5, 3)
print(x)
# All row with first column
print(x[:, 0])
# All column with first row
print(x[0, :])
# One item
print(x[0, 0])
# Get single item's value
print(x[0, 0].item())


# 【Part IV: Reshape】
x1 = torch.rand(4, 4)
# reshape to 1-D
print(x1.view(16))
# reshape with auto calculation
print(x1.view(-1, 8))  # the second dim is set to 8, the first dim is calculated accordingly


# 【Part V: tensor <-> numpy】
# tensor -> numpy
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))
# Cautions1: if these data are running in cpu, they store in the same place,
# if we change one, another will be modified together.
a.add_(1)
print(b)

# numpy -> tensor
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a += 1
print(b)
# Still same with the Cautions1

if torch.cuda.is_available():
    print('CUDA available')
    device = torch.device("cuda")
    x = torch.ones(5, device=device)  # create a tensor in GPU
    y = torch.ones(5)
    y = y.to(device)  # move y in cpu to gpu
    z = x + y
    # z.numpy()  # bug, cannot convert data in gpu, because numpy can only deal with data in cpu
    z = z.to("cpu")  # correct


# 【Part VI: grad related】
# requires_grad is False in default
x = torch.ones(5, requires_grad=True)
print(x)