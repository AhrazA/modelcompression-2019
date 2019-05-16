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
from configurations import configurations

def main():
    parser = argparse.ArgumentParser(description='Prunes a network.')

    parser.add_argument('--config', type=str, required=True, metavar="C",
                        help="Name of the configuration in configurations.py to run.")

    parser.add_argument('--pretrained-weights', type=str,
                        help="Path of the pretrained weights to load.")
    
    parser.add_argument('--tensorboard', action='store_true', help="Enable tensorboard logging")

    setup_default_args(parser)

    args = parser.parse_args()

    config = [x for x in configurations if x['name'] == args.config]

    if len(config) != 1:
        raise ValueError("Invalid configuration parameter.")
    
    config = config[0]
    
    device = 'cpu' if args.no_cuda else 'cuda:0'

    if args.tensorboard:
        writer = SummaryWriter()
    
    train_data = config['dataset'](
        './data', train=True, download=True, transform=transforms.Compose(config['transforms'])
    )

    test_data = config['dataset'](
        './data', train=False, download=True, transform=transforms.Compose(config['transforms'])
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, num_workers=1)

    N = len(train_loader)
    l = 1e-4
    wr = l**2. / N
    dr = 2. / N

    model = config['model'](wr, dr).to(device)

    optimizer = config['optimizer'](model.parameters(), lr=args.lr, momentum=args.momentum)

    wrapper = Classifier(model, device, train_loader, test_loader)
    state_dict = model.state_dict(torch.load(args.pretrained_weights, map_location=torch.device(device)))
    model.load_state_dict(state_dict)

    wrapper.train(10, optimizer, 5, config['loss_fn'])
    final_loss, all_outputs = wrapper.test(config['loss_fn'], multiple_pred=True)
    means = [torch.mean(x) for x in all_outputs]

    print(final_loss)
    print('Means: ', means)


if __name__ == '__main__':
    main()