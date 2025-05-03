import torch
import torch.nn as nn
from collections import OrderedDict

class Network(nn.Module):
    def __init__(
        self,
        input_size, 
        hidden_size,
        output_size, 
        depth, 
        act=torch.nn.Tanh, 
    ):
        super(Network, self).__init__()

        layers = [('input', torch.nn.Linear(input_size, hidden_size))]
        layers.append(('input_activation', act()))

        for i in range(depth):
            layers.append(
                ('hidden_%d' % i, torch.nn.Linear(hidden_size, hidden_size))
            )
            layers.append(('activation_%d' % i, act()))

        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))

        self.layers = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)

