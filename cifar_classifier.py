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


class MaskedCifar(nn.Module):
    def __init__(self):
        super(MaskedCifar, self).__init__()
        self.conv1 = MaskedConv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = MaskedConv2d(6, 16, 5)
        self.fc1 = MaskedLinear(16 * 5 * 5, 120)
        self.fc2 = MaskedLinear(120, 84)
        self.fc3 = MaskedLinear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def set_masks(self, masks):
        self.conv1.set_mask(masks[0])
        self.conv2.set_mask(masks[1])
        self.fc1.set_mask(masks[2])
        self.fc2.set_mask(masks[3])
        self.fc3.set_mask(masks[4])

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    setup_default_args(parser)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if not args.no_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    classifier = Classifier(MaskedCifar(), 'cuda', train_loader, test_loader)
    optimizer = optim.SGD(classifier.model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        classifier.train(args.log_interval, optimizer, epoch, F.cross_entropy)
        classifier.test(F.cross_entropy)
    
    if (args.save_model):
        torch.save(classifier.model.state_dict(),"models/cifar_classifier.pt")

if __name__ == '__main__':
    main()
