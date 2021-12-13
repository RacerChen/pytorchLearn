# Learning logistics regression in pytorch
# logistics regression is similar with linear regression,
# it adds a logistics function(sigmoid) to convert it to a classification task
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape  # 569 30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale （归一化）
'''
做归一化的原因：
* 归一化加快了梯度下降求最优解的速度
    如果机器学习模型使用了地图下降法求最优解时，归一化往往非常必要，否则很难收敛，甚至不收敛
* 归一化可能提高精度
    一些分类器需要计算样本之间的距离（如欧氏距离），如果一个特征值范围非常大，那么距离的计算就主要取决于这个特征，从而与实际情况相悖。
当然，也有部分模型是不需要做归一化的，比如概率模型（树形模型），因为他们不关心变量的值，而是关系变量的分布和变量之间的条件概率，如决策树，随机森林
StandartScaler是一个去均值和方差归一化。且是针对每一个特征维度来做的。
x* = (x-mean) / std

具体函数：
fit_transform： 选取部分数据计算统计指标，例如均值、方差、最大最小值等，然后利用这些值进行归一化
transform： 在前面fit后的基础上，进行归一化
'''
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)  # transform in one column
y_test = y_test.view(y_test.shape[0], 1)  # transform in one column


# 1) model
# f = wx + b, sigmoid function at the end
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward
    y_pred = model(X_train)

    # loss
    loss = criterion(y_pred, y_train)

    # backward
    loss.backward()

    # updates
    optimizer.step()

    # empty grad
    optimizer.zero_grad()

    if epoch % 5 == 0:
        print(f'epoch:{epoch}, loss:{loss.item():.4f}')


with torch.no_grad():
    y_predicted = model(X_test)

    y_predicted_class = y_predicted.round()

    acc = y_predicted_class.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy={acc:.4f}')


[w, b] = model.parameters()
print(w)
print(b)
# 1 linear layer, is not a deep model, but still reach Accuracy=0.9561
