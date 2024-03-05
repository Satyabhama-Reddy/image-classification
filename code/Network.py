### YOUR CODE HERE
import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class DenseNet(nn.Module):
    def __init__(self, configs):
        
        super(DenseNet, self).__init__()
        # print(configs)
        self.configs = configs
        self.drop_rate = configs['drop_rate']

        growth_rate = configs['growth_rate'] # used in dense layers
        num_blocks = configs['num_blocks']  # list of number of bottleneck blocks in each dense layer
        reduction = configs['reduction'] # used in transition layers
        num_classes = configs['num_classes']

        layers = []
        num_planes = 2 * growth_rate
        layers.append(Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)) # Initial convolution layer
        for i in range(3): #First 3 dense and transition blocks
            layers.append(self._make_dense_layers(BottleneckBlock, num_planes, num_blocks[i], growth_rate))
            layers.append(nn.Dropout(p=self.drop_rate))  #Dropout for regularization
            num_planes += num_blocks[i] * growth_rate
            filters_out = int(math.floor(num_planes * reduction))
            layers.append(TransitionBlock(num_planes, filters_out))
            num_planes = filters_out

        layers.append(self._make_dense_layers(BottleneckBlock, num_planes, num_blocks[3], growth_rate)) #4th dense block
        num_planes += num_blocks[3] * growth_rate
        layers.append(nn.BatchNorm2d(num_planes))
        layers.append(nn.ReLU())
        layers.append(nn.AvgPool2d(4))

        self.dense_network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_planes, num_classes)
        self.dropout = nn.Dropout(p=self.drop_rate)

    def _make_dense_layers(self, block, filters_in, num_blocks, growth_rate):
        layers = []
        for i in range(num_blocks):
            layers.append(block(filters_in, growth_rate))
            filters_in += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.dense_network(x)
        output = output.view(output.size(0), -1)
        output = self.dropout(self.fc(output)) #Dropout for regularization
        return output

class BottleneckBlock(nn.Module):
    def __init__(self, filters_in, growth_rate):
        super(BottleneckBlock, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(filters_in),
            nn.ReLU(),
            Conv2d(filters_in, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(),
            Conv2d(4 * growth_rate, growth_rate, kernel_size=3, bias=False, padding=1)
        )

    def forward(self, x):
        output = self.bottleneck(x)
        output = torch.cat([output, x], 1)  # concatenating feature maps, essense of DenseNet
        return output


class TransitionBlock(nn.Module):
    def __init__(self, filters_in, filters_out):
        super(TransitionBlock, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(filters_in),
            nn.ReLU(),
            Conv2d(filters_in, filters_out, kernel_size=1, bias=False),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.transition(x)


# Introduced to incorporate weight standardization.
class Conv2d(nn.Conv2d):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(channels_in, channels_out, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        # Weight Standardization
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

### END CODE HERE

### END CODE HERE
