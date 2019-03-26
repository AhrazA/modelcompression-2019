from __future__ import print_function

import json
import os
import numpy as np
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms

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

def yolo_config(config, args):
    # if (args.prune_threshold > 0.005):
    #     print("WARNING: Prune threshold seems too large.")
    #     if input("Input y if you are sure you want to continue.") != 'y': return

    model = config['model'](config['config_path'])
    device = 'cpu' if args.no_cuda else 'cuda:1'
    wrapper = YoloWrapper(device, model)
    lr0 = 0.001
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

    print("Loading dataloaders..")
    train_dataloader = LoadImagesAndLabels(config['datasets']['train'], batch_size=args.batch_size, img_size=config['image_size'])
    val_dataloader = LoadImagesAndLabels(config['datasets']['val'] if config['val_set_for_train'] == 'val' else config['datasets']['test'], batch_size=args.batch_size, img_size=config['image_size'])

    model.to(device)

    if (args.pretrained_weights):
        print("Loading pretrained weights..")
        model.load_state_dict(torch.load(args.pretrained_weights)['model'])
    else:
        wrapper.train(train_dataloader, val_dataloader, args.epochs, optimizer, lr0)

    with torch.no_grad():
        pre_prune_mAP, _, _  = wrapper.test(val_dataloader, img_size=config['image_size'], batch_size=args.batch_size)

    prune_perc = 0. if args.start_at_prune_rate is None else args.start_at_prune_rate
    prune_iter = 0
    curr_mAP = pre_prune_mAP

    while (curr_mAP - pre_prune_mAP) > -args.prune_threshold:
        prune_iter += 1
        prune_perc += 5.
        masks = weight_prune(model, prune_perc)
        model.set_mask(masks)

        if not args.no_retrain:
            print(f"Retraining at prune percentage {prune_perc}..")
            curr_mAP, best_weights = wrapper.train(train_dataloader, val_dataloader, 3, optimizer, lr0)

            print("Loading best weights from training epochs..")
            model.load_state_dict(best_weights)
        else:
            with torch.no_grad():
                curr_mAP, _, _ = wrapper.test(val_dataloader, img_size=config['image_size'], batch_size=args.batch_size)

        print(f"mAP achieved: {curr_mAP}")
        print(f"Change in mAP: {curr_mAP - pre_prune_mAP}")

    prune_perc = prune_rate(model)

    if (args.save_model):
        torch.save(model.state_dict(), f'{config["name"]}-pruned-{datetime.datetime.now()}')

    print(f"Pruned model: {config['name']}")
    print(f"Pre-pruning mAP: {pre_prune_mAP}")
    print(f"Post-pruning mAP: {curr_mAP}")
    print(f"Percentage of zeroes: {prune_perc}")

def classifier_config(config, args):
    model = config['model']()
    device = 'cpu' if args.no_cuda else 'cuda'

    train_data = test_data = config['dataset'](
        './data', train=True, download=True, transform=transforms.Compose(config['transforms'])
    )

    test_data = config['dataset'](
        './data', train=False, download=True, transform=transforms.Compose(config['transforms'])
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    optimizer = config['optimizer'](model.parameters(), lr=args.lr, momentum=args.momentum)
    
    wrapper = Classifier(model, 'cuda', train_loader, test_loader)

    if (args.pretrained_weights):
        print("Loading pretrained weights..")
        model.load_state_dict(torch.load(args.pretrained_weights))
    else:
        for epoch in range(1, args.epochs + 1):
            wrapper.train(args.log_interval, optimizer, epoch, config['loss_fn'])
    
    pre_prune_accuracy = wrapper.test(config['loss_fn'])    
    prune_perc = 0. if args.start_at_prune_rate is None else args.start_at_prune_rate
    prune_iter = 0
    curr_accuracy = pre_prune_accuracy

    while (pre_prune_accuracy - curr_accuracy) < args.prune_threshold:
        prune_iter += 1
        prune_perc += 5.
        masks = weight_prune(model, prune_perc)
        model.set_mask(masks)

        print(f"Testing at prune percentage {prune_perc}..")
        curr_accuracy = wrapper.test(config["loss_fn"])

        print(f"Accuracy achieved: {curr_accuracy}")
        print(f"Change in accuracy: {pre_prune_accuracy - curr_accuracy}")

        if args.no_retrain: continue
        
        for epoch in range(1, args.epochs + 1):
            wrapper.train(args.log_interval, optimizer, epoch, config['loss_fn'])

    prune_perc = prune_rate(model)    

    if (args.save_model):
        torch.save(model.state_dict(), f'{config["name"]}-pruned-{datetime.datetime.now()}')

    print(f"Pruned model: {config['name']}")
    print(f"Pre-pruning accuracy: {pre_prune_accuracy}")
    print(f"Post-pruning accuracy: {curr_accuracy}")
    print(f"Percentage of zeroes: {prune_perc}")

def main():
    parser = argparse.ArgumentParser(description='Test Bench')

    parser.add_argument('--config', type=str, required=True, metavar="C",
                        help="Name of the configuration in configurations.py to run.")

    parser.add_argument('--pretrained-weights', type=str,
                        help="Path of the pretrained weights to load.")
    
    parser.add_argument('--prune-threshold', type=float, default=0.05,
                        help='The accuracy threshold at which to stop pruning.')

    parser.add_argument('--no-retrain', action='store_true',
                        help="Do not retrain after pruning.")

    parser.add_argument('--start-at-prune-rate', type=float, help="The percentage to begin pruning at. Default: 0")

    setup_default_args(parser)
    args = parser.parse_args()
    print(args)
    
    chosen_config = [x for x in configurations if x['name'] == args.config]
    
    if len(chosen_config) != 1:
        raise ValueError("Invalid configuration parameter.")
    
    if chosen_config[0]['type'] == 'classifier':
        classifier_config(chosen_config[0], args)
    
    if chosen_config[0]['type'] == 'yolo':
        yolo_config(chosen_config[0], args)

if __name__ == '__main__':
    main()
