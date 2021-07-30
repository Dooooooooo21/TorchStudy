#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/5/22 11:40
# @Author  : dly
# @File    : base_operator.py
# @Desc    :

import torch
from torch import nn
from torch import optim
from torch.nn import init


def tensor_exe():
    x = torch.empty(5, 3)
    print(x)

    print(x.size())
    print(x.shape)


# 梯度
def grad_exe():
    y = torch.rand(2, 2, requires_grad=True)
    print(y)
    print(y.grad_fn)

    z = y + 2
    print(z)
    print(z.grad_fn)

    print(y.is_leaf, z.is_leaf)


# 网络
def net_exe():
    net = nn.Sequential()
    net.add_module('linear', nn.Linear(6, 1))

    print(net)
    print('-' * 20)

    # 参数
    print(net.parameters())
    for para in net.parameters():
        print(para)

    # 优化器
    optimizer = optim.SGD(net.parameters(), lr=0.03)
    print(optimizer)


def mlp():
    """
    多层感知机 MLP
    隐藏层、激活函数、
    """
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    net = nn.Sequential(
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs),
    )

    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)


if __name__ == '__main__':
    net_exe()
