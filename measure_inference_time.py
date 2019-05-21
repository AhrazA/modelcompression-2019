import argparse
import datetime

import torch
import torch.optim as optim
import time
from torchvision import datasets, transforms

from cifar_classifier import MaskedCifar
from classifier import Classifier
from mnist_classifier import MaskedMNist
from pruning.methods import weight_prune, prune_rate, get_all_weights, quantize_k_means
from pruning.utils import to_var
from resnet import MaskedResNet18, MaskedResNet34, MaskedResNet50, MaskedResNet101, MaskedResNet152
from classifier_utils import setup_default_args
from yolov3 import LoadImagesAndLabels, YoloWrapper
from fasterrcnn.wrapper import FasterRCNNWrapper

from torch.utils.tensorboard import SummaryWriter
from configurations import configurations

def yolo_config(config, args):
    device = 'cpu' if args.no_cuda else 'cuda:1'
    model = config['model'](config['config_path'], device=device)
    wrapper = YoloWrapper(device, model)
    lr0 = 0.001
    # lr0 = args.lr
    optimizer = config['optimizer'](filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=args.momentum)
    writer = SummaryWriter()

    print("Loading dataloaders..")
    train_dataloader = LoadImagesAndLabels(config['datasets']['train'], batch_size=args.batch_size, img_size=config['image_size'])
    val_dataloader = LoadImagesAndLabels(config['datasets']['test'], batch_size=args.batch_size, img_size=config['image_size'])

    print("Loading pretrained weights..")

    print("Measuring inference time..")

    start_time = time.time()
    
    with torch.no_grad():
        pre_prune_mAP, _, _  = wrapper.test(val_dataloader, img_size=config['image_size'], batch_size=args.batch_size)

    end_time = time.time()

    return wrapper

def classifier_config(config, args):
    model = config['model']()
    device = 'cpu' if args.no_cuda else 'cuda:0'
    
    if args.tensorboard:
        writer = SummaryWriter()

    train_data = test_data = config['dataset'](
        './data', train=True, download=True, transform=transforms.Compose(config['transforms'])
    )

    test_data = config['dataset'](
        './data', train=False, download=True, transform=transforms.Compose(config['transforms'])
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, num_workers=1)
    optimizer = config['optimizer'](model.parameters(), lr=args.lr, momentum=args.momentum)
    
    wrapper = Classifier(model, device, train_loader, test_loader)

    print("Loading pretrained weights..")
    model.load_state_dict(torch.load(args.pretrained_weights, map_location=torch.device(device)))
    
    start_time = time.time()

    for i in range(args.repititions):
        with torch.no_grad():
            curr_accuracy = wrapper.test(config['loss_fn'])
    
    end_time = time.time()

    print("Average inference time: ", (end_time - start_time) / args.repititions)

    return wrapper

def frcnn_config(config, args):
    classes = ('__background__',  # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')

    model = config['model'](
        classes
        # model_path = args.pretrained_weights
    )

    model.create_architecture()

    wrapper = FasterRCNNWrapper('cpu' if args.no_cuda else 'cuda:0', model)

    if args.tensorboard:
        writer = SummaryWriter()

    print("Loading weights ", args.pretrained_weights)
    state_dict = torch.load(args.pretrained_weights, map_location=torch.device('cuda:0'))
        
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
        
    model.load_state_dict(state_dict)

    start_time = time.time()

    wrapper.test()

    end_time = time.time()

    return wrapper

def main():
    parser = argparse.ArgumentParser(description='Prunes a network.')

    parser.add_argument('--config', type=str, required=True, metavar="C",
                        help="Name of the configuration in configurations.py to run.")

    parser.add_argument('--pretrained-weights', type=str, required=True,
                        help="Path of the pretrained weights to load.")

    parser.add_argument('--tensorboard', action='store_true', help="Enable tensorboard logging")

    parser.add_argument('--repititions', type=int, default=10)

    setup_default_args(parser)
    args = parser.parse_args()
    
    chosen_config = [x for x in configurations if x['name'] == args.config]
    
    if len(chosen_config) != 1:
        raise ValueError("Invalid configuration parameter.")
    
    if chosen_config[0]['type'] == 'classifier':
        wrapper = classifier_config(chosen_config[0], args)
    
    if chosen_config[0]['type'] == 'yolo':
        wrapper = yolo_config(chosen_config[0], args)
    
    if chosen_config[0]['type'] == 'frcnn':
        wrapper = frcnn_config(chosen_config[0], args)
    
if __name__ == '__main__':
    main()
