#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/7/31 0:32
# @Author  : dly
# @File    : create_net.py
# @Desc    : 模型构造
import torch
from torch import nn


# 继承 Module
class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)  # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


"""
1. Sequential
2. ModuleList
3. ModuleDict
"""

"""
init 模块初始化参数
1. init.normal_
2. init.constant_
3. 自定义初始化函数
4. 多层之间共享参数
5. 模型参数的延后初始化, 在坐前向net(x)运算时，进行初始化操作
"""

"""
自定义层
继承nn.Module, 写__init__, forward函数
"""

"""
model 保存、读取
save、load
1. torch.save(model.state_dict(), PATH) # 推荐的文件后缀名是pt或pth
2. torch.save(model, PATH)
"""

if __name__ == '__main__':
    X = torch.rand(2, 784)
    net = MLP()
    print(net)

    print(net.state_dict())

    out = net(X)
    print(out)
