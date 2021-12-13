# Tutorial from: https://www.youtube.com/watch?v=DbeIqrwb_dE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=3
import torch

# uniform distribution
x = torch.randn(3, requires_grad=True)
# print(x)
# standard normal  distribution
# x = torch.rand(3)

y = x+2
# print(y)
# output: tensor([1.7933, 1.9224, 3.0957], grad_fn=<AddBackward0>)

z = y*y*2
# print(z)
# output: tensor([ 2.2092, 15.5164,  7.2589], grad_fn=<MulBackward0>)

z = z.mean()
# print(z)
# output: tensor(3.2330, grad_fn=<MeanBackward0>)

z.backward()  # dz/dx
# print(x.grad)

a = y*y*2
# a.backward()
# bug error: RuntimeError: grad can be implicitly created only for scalar outputs
# Using mean, a vector z is transform to scalar

# Question1: How to generate backward of vector?
# Solution: add parameter jacobian matrix（雅可比矩阵） to do jacobian-vector product
'''
原理：
上述情况是由于Tensor的Element-wise运算机制导致的，比如X=[x1,x2,x2]
eg1.Y = X^2, Y = [x1^2, x2^2, x3^2]
因为XY是向量而不是实数，所以dY/dX != 2X,而是一个雅可比矩阵j
     dy1/dx1, dy1/dx2, dy1/dx3     2x1, 0, 0
J = (dy2/dx1, dy2/dx2, dy2/dx3) = (0, 2x2, 0)
     dy3/dx1, dy3/dx2, dy3/dx3     0, 0, 2x3
backward的到J,需要乘以一个投影向量v，才能得到参数到各个x上的反向传播偏导
        dy1/dx1 ··· dym/dxz     dl/dy1       dl/dx1   
J·v = ( ·         ·       · ) (   ·    ) = (   ·    )
        dy1/dxn ··· dym/dxn     dl/dym       dl/dxn
相当于:f(x1,x2,···xn) = (y1(x1,x2,···xn), y2(x1,x2,···xn), ··· ,ym(x1,x2,···xn))
最终将n维度空间映射到m维上
对于例子eg1，v=(1,1,1)
反向传播结果为 (2x1, 2x2, 2x3)
'''
b = torch.ones(3, requires_grad=True)
c = b*b
c.backward(torch.tensor([1, 1, 1], dtype=torch.float64))
# print(b.grad)
# output: tensor([2., 2., 2.])


# Cancel grad during runtime
'''
1 v.requires_grad_(False)
2 v.detach()
3 with torch.no_grad():
'''
v = torch.randn(3, requires_grad=True)
# print(v)
# cancel it with
v.requires_grad_(False)
# print(v)
v.detach()
# print(v)
with torch.no_grad():
    v_no = v+2
    # print(v_no)
# print(v_no)

# Caution: grad will accumulate with the training epochs, for example
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()

# output:
'''
Obviously, it's false
tensor([3., 3., 3., 3.])
tensor([6., 6., 6., 6.])
tensor([9., 9., 9., 9.])
so we need add zero_(), to refresh it, now:
tensor([3., 3., 3., 3.])
tensor([3., 3., 3., 3.])
tensor([3., 3., 3., 3.])

# it is same when we use torch optimizer
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()

'''

