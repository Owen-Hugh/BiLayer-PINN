import torch
import torch.nn as nn
from collections import OrderedDict

# 定义神经网络的架构
class Network(nn.Module):
    # 构造函数
    def __init__(
        self,
        input_size, # 输入层神经元数
        hidden_size, # 隐藏层神经元数
        output_size, # 输出层神经元数
        depth, # 隐藏层数
        act=torch.nn.Tanh, # 输入层和隐藏层的激活函数
    ):
        super(Network, self).__init__()#调用父类的构造函数

        # 输入层
        layers = [('input', torch.nn.Linear(input_size, hidden_size))]
        layers.append(('input_activation', act()))

        # 隐藏层
        for i in range(depth):
            layers.append(
                ('hidden_%d' % i, torch.nn.Linear(hidden_size, hidden_size))
            )
            layers.append(('activation_%d' % i, act()))

        # 输出层
        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))

        #将这些层组装为神经网络
        self.layers = torch.nn.Sequential(OrderedDict(layers))

    # 前向计算方法
    def forward(self, x):
        return self.layers(x)

