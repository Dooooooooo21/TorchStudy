#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2023/8/9 20:44
# @Author  : dly
# @File    : code_test.py
# @Desc    :
import torch

x = torch.arange(10, dtype=float, requires_grad=True)
y = 2 * torch.dot(x, x)
y.backward()
print(x.grad)
print(x)
