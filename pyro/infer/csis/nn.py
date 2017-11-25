from __future__ import absolute_import, division, print_function

import pyro

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Artifact(nn.module):
    def __init__(self):
        super(Artifact, self).__init__()
        self.pool = nn.MaxPool2d(5, stride=5)
        self.conv = nn.Conv2d(3, 1, 1)  # blends the three RGB layers together
        self.fcn1 = nn.Linear(1600, 10)
        self.fcn2 = nn.Linear(10, 1)

    def forward(self, images):
        x = images.view(-1, 3, 200, 200)
        x = self.pool(x)
        x = F.relu(self.conv(x))
        x = x.view(-1, 1600)
        x = F.relu(self.fcn1(x))
        x = self.fcn2(x)
        sig = nn.Sigmoid()
        x = 10 * sig(x)     # some may say that this makes it very specific to the problem at hand
        x = x.view(-1)
        return x
