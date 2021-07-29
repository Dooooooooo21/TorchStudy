#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/5/22 11:40
# @Author  : dly
# @File    : base_operator.py
# @Desc    :

import torch

x = torch.empty(5, 3)
print(x)

print(x.size())
print(x.shape)

# 梯度
y = torch.rand(2, 2, requires_grad=True)
print(y)
print(y.grad_fn)

z = y + 2
print(z)
print(z.grad_fn)

print(y.is_leaf, z.is_leaf)
