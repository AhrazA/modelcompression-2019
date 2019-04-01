import argparse
import datetime

import torch
import torch.optim as optim
from torchvision import datasets, transforms

from cifar_classifier import MaskedCifar
from classifier import Classifier
from mnist_classifier import MaskedMNist
from pruning.methods import weight_prune, prune_rate, get_all_weights, quantize_k_means
from pruning.utils import to_var
from resnet import MaskedResNet18, MaskedResNet34, MaskedResNet50, MaskedResNet101, MaskedResNet152
from classifier_utils import setup_default_args
from yolov3 import LoadImagesAndLabels, YoloWrapper

from tensorboardX import SummaryWriter

from configurations import configurations

if __name__ == '__main__':
    config = [x for x in configurations if x['name'] == 'FCCifar10Classifier'][0]

    model = config['model']()

    device = 'cuda'

    train_data = test_data = config['dataset'](
        './data', train=True, download=True, transform=transforms.Compose(config['transforms'])
    )

    test_data = config['dataset'](
        './data', train=False, download=True, transform=transforms.Compose(config['transforms'])
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
    optimizer = config['optimizer'](model.parameters(), lr=0.01, momentum=0.5)
    
    wrapper = Classifier(model, 'cuda', train_loader, test_loader)

    model.load_state_dict(torch.load('./models/cifar_classifier.pt'))

    print("Started quantizing")

    quantize_k_means(model)
    
    print("Finished quantizing")

    optimizer = config['optimizer'](model.parameters(), lr=0.01, momentum=0.5)
    
    wrapper.train(10, optimizer, 1, config['loss_fn'])
