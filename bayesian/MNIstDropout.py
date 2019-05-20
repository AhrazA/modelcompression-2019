from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
from classifier import Classifier
from classifier_utils import setup_default_args
from bayesian.ConcreteDropoutLinear import ConcreteDropoutLinear, ConcreteDropoutConvolutional
from bayesian.utils import heteroscedastic_loss, logsumexp
from mnist_classifier import MaskedMNist
from cifar_classifier import MaskedCifar
from classifier import Classifier
from torch.utils.tensorboard import SummaryWriter
import configurations

class MaskedConcreteCifar(MaskedCifar):
    def __init__(self, weight_regularizer, dropout_regularizer, **kwargs):
        super(MaskedConcreteCifar, self).__init__()

class MaskedConcreteMNist(MaskedMNist):
    def __init__(self, weight_regularizer, dropout_regularizer, **kwargs):
        super(MaskedConcreteMNist, self).__init__()

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

        self.conv_drop1 = ConcreteDropoutConvolutional(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer, temp = 2. / 3.)
        self.conv_drop2 = ConcreteDropoutConvolutional(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer, temp = 2. / 3.)

        self.conc_drop1 = ConcreteDropoutLinear(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop2 = ConcreteDropoutLinear(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)

        self.conc_drop_mu = ConcreteDropoutLinear(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop_logvar = ConcreteDropoutLinear(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)

        self.linear_mu = nn.Linear(10, 1)
        self.linear_logvar = nn.Linear(10, 1)

    def forward(self, x):
        regularization = torch.empty(6, device=x.device)

        x, regularization[0] = self.conv_drop1(x, nn.Sequential(self.conv1, self.relu))
        x = F.max_pool2d(x, 2, 2)

        x, regularization[1] = self.conv_drop2(x, nn.Sequential(self.conv2, self.relu))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4*4*50)

        x, regularization[2] = self.conc_drop1(x, nn.Sequential(self.fc1, self.relu))
        x, regularization[3] = self.conc_drop2(x, nn.Sequential(self.fc2, self.softmax))

        # mean, regularization[4] = self.conc_drop_mu(x, self.linear_mu)
        # log_var, regularization[5] = self.conc_drop_logvar(x, self.linear_logvar)

        # return mean, log_var, regularization.sum()
        return x

if __name__ == '__main__':
    device = 'cuda:0'

    train_data = test_data = datasets.MNIST(
        './data', train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
    )

    test_data = datasets.MNIST(
        './data', train=False, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=1)

    pre_trained_weights = torch.load('./models/mnist_classifier.pt', map_location=torch.device(device))

    model = MaskedConcreteMNist(wr, dr).to(device)

    state_dict = model.state_dict()
    state_dict.update(pre_trained_weights)
    model.load_state_dict(state_dict)

    wrapper = Classifier(model, 'cuda:0', train_loader, test_loader)
    wrapper.train(10, torch.optim.Adadelta(model.parameters()), 5, F.cross_entropy)
    wrapper.test(F.cross_entropy)