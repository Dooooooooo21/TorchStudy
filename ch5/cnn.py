#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/8/1 16:37
# @Author  : dly
# @File    : cnn.py
# @Desc    : 卷积

import torch
from torch import nn

"""
1. 卷积计算
2. 特征图、感受野
3. 填充、步幅
4. 多输入通道, 1 * 1 卷积
5. 池化
6. LeNet, AlexNet, VGG
"""

"""
1. 堆积的小卷积核优于大的卷积核, 可以增加深度
2. Inception, 并行连接
3. BN
4. ResNet
5. DenseNet
"""


# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])  # 排除不关心的前两维：批量和通道


if __name__ == '__main__':
    # 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

    X = torch.rand(8, 8)
    print(comp_conv2d(conv2d, X).shape)