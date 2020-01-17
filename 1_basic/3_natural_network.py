
# 一个神经网络的典型训练过程如下
# 1、定义神经网络（包含一些可学习参数/权重）
# 2、在输入数据集上迭代
# 3、通过网络处理输入
# 4、计算损失（输出和正确答案的距离）
# 5、将梯度反向传播给网络参数
# 6、跟新网络权重（weight = weight - learning_rate * gradient）

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
params = list(net.parameters())
print(len(params))
print(params[9].size())


input = torch.randn(1, 1, 32, 32)
output = net(input)
print(output)


# 损失函数
criterion = nn.MSELoss()


# 反向传播
net.zero_grad()     # 清零所有参数的梯度缓存

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# 更新权重
# 方法1：简单的SGD
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# 方法2：使用内置的优化方法
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()   # 清零梯度缓存
output = net(input)
loss = loss = criterion(output, target)
loss.backward()
optimizer.step()

