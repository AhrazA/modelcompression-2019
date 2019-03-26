from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from classifier import Classifier
from classifier_utils import setup_default_args
from pruning.masked_conv_2d import MaskedConv2d
from pruning.masked_linear import MaskedLinear

class MaskedMNist(nn.Module):
    def __init__(self):
        super(MaskedMNist, self).__init__()
        self.conv1 = MaskedConv2d(1, 20, 5, 1)
        self.conv2 = MaskedConv2d(20, 50, 5, 1)
        self.fc1 = MaskedLinear(4*4*50, 500)
        self.fc2 = MaskedLinear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def set_mask(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.conv1.set_mask(masks[0])
        self.conv2.set_mask(masks[1])
        self.fc1.set_mask(masks[2])
        self.fc2.set_mask(masks[3])
