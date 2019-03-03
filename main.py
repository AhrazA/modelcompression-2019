from __future__ import print_function

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from cifar_classifier import MaskedCifar
from classifier import Classifier
from mnist_classifier import MaskedMNist
from pruning.methods import weight_prune, prune_rate
from pruning.utils import to_var


def get_list_choice(choices):
    for i, m in enumerate(choices):
        print("\t{}:{}".format(i, m))
    
    choice = int(input("Enter selected index and press enter: "))
    
    if choice < 0 or choice > len(choices):
        raise ValueError("Index out of range.")
    
    return choice

def load_model(model_file_name, configuration):
    model = configuration['model']()
    model.load_state_dict(torch.load('./models/' + model_file_name))

    test_data = configuration['dataset'](
        './data', train=False, transform=transforms.Compose(configuration['transforms'])
    )

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=True, num_workers=1, pin_memory=True)

    return Classifier(model, 'cuda', None, test_loader)

def main():
    configurations = [
        {
            'model': MaskedCifar,
            'dataset': datasets.CIFAR10,
            'transforms': 
                        [
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ],
            'loss_fn': F.cross_entropy
        },
        {
            'model': MaskedMNist,
            'dataset': datasets.MNIST,
            'transforms':
                        [
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ],
            'loss_fn': F.nll_loss
        }
    ]

    print("Select a model type to prune. Models available:")
    model_choice = get_list_choice(configurations)

    print()

    print("Select a model params file to load by index. Models available:")
    saved_models = os.listdir('./models/')
    file_choice = get_list_choice(saved_models)

    print()

    prune_perc = float(input("Select pruning percentage: (0-100)%: "))
    if prune_perc < 0 or prune_perc > 100.0:
        raise ValueError("Pruning percentage be a percentage value between 0 and 100.")

    model_file_name = saved_models[file_choice]
    chosen_configuration = configurations[model_choice]
    print("Loading file {} for model {}".format(model_file_name, chosen_configuration))
    wrapped_model = load_model(model_file_name, chosen_configuration)

    print()

    print("Testing pre-pruned model..")
    wrapped_model.test(chosen_configuration["loss_fn"])

    print("Pruning model..")
    masks = weight_prune(wrapped_model.model, prune_perc)
    wrapped_model.model.set_masks(masks)

    prune_rate(wrapped_model.model)

    print()

    print("Evaluating pruned model..")
    wrapped_model.test(chosen_configuration["loss_fn"])

if __name__ == '__main__':
    main()
