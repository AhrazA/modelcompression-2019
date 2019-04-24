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

def yolo_config(config, args):
    # if (args.prune_threshold > 0.005):
    #     print("WARNING: Prune threshold seems too large.")
    #     if input("Input y if you are sure you want to continue.") != 'y': return

    device = 'cpu' if args.no_cuda else 'cuda:3'
    model = config['model'](config['config_path'], device=device)
    wrapper = YoloWrapper(device, model)
    lr0 = 0.001
    optimizer = config['optimizer'](filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=args.momentum)

    print("Loading dataloaders..")
    train_dataloader = LoadImagesAndLabels(config['datasets']['train'], batch_size=args.batch_size, img_size=config['image_size'])
    val_dataloader = LoadImagesAndLabels(config['datasets']['test'], batch_size=args.batch_size, img_size=config['image_size'])

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
        torch.save(model.state_dict(), f'{config["name"]}-pruned-{datetime.datetime.now().strftime("%Y%m%d%H%M")}.pt')

    print(f"Pruned model: {config['name']}")
    print(f"Pre-pruning mAP: {pre_prune_mAP}")
    print(f"Post-pruning mAP: {curr_mAP}")
    print(f"Percentage of zeroes: {prune_perc}")

    return wrapper

def classifier_config(config, args):
    model = config['model']()
    device = 'cpu' if args.no_cuda else 'cuda:0'

    train_data = test_data = config['dataset'](
        './data', train=True, download=True, transform=transforms.Compose(config['transforms'])
    )

    test_data = config['dataset'](
        './data', train=False, download=True, transform=transforms.Compose(config['transforms'])
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    optimizer = config['optimizer'](model.parameters(), lr=args.lr, momentum=args.momentum)
    
    wrapper = Classifier(model, device, train_loader, test_loader)

    if (args.pretrained_weights):
        print("Loading pretrained weights..")
        model.load_state_dict(torch.load(args.pretrained_weights))
    else:
        wrapper.train(args.log_interval, optimizer, args.epochs, config['loss_fn'])
    
    pre_prune_accuracy = wrapper.test(config['loss_fn'])    
    prune_perc = 0. if args.start_at_prune_rate is None else args.start_at_prune_rate
    prune_iter = 0
    curr_accuracy = pre_prune_accuracy

    while (curr_accuracy - pre_prune_accuracy) > -args.prune_threshold:
        print(f"Testing at prune percentage {prune_perc}..")
        curr_accuracy = wrapper.test(config["loss_fn"])

        print(f"Accuracy achieved: {curr_accuracy}")
        print(f"Change in accuracy: {pre_prune_accuracy - curr_accuracy}")
        
        prune_iter += 1
        prune_perc += 5.
        masks = weight_prune(model, prune_perc)
        model.set_mask(masks)

        if not args.no_retrain:
            print(f"Retraining at prune percentage {prune_perc}..")
            curr_accuracy, best_weights = wrapper.train(args.log_interval, optimizer, args.epochs, config['loss_fn'])

            print("Loading best weights from training epochs..")
            model.load_state_dict(best_weights)
        else:
            with torch.no_grad():
                curr_accuracy, _, _ = wrapper.test(config['loss_fn'])
    
    prune_perc = prune_rate(model)    

    if (args.save_model):
        torch.save(model.state_dict(), f'./models/{config["name"]}-pruned-{datetime.datetime.now().strftime("%Y%m%d%H%M")}.pt')

    print(f"Pruned model: {config['name']}")
    print(f"Pre-pruning accuracy: {pre_prune_accuracy}")
    print(f"Post-pruning accuracy: {curr_accuracy}")
    print(f"Percentage of zeroes: {prune_perc}")

    return wrapper

def frcnn_config(config, args):
    classes = ('__background__',  # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')

    print(args.pretrained_weights)

    model = config['model'](
        classes,
        model_path = args.pretrained_weights
    )

    model.create_architecture()
    pre_prune_accuracy = 

    # masks = pruning.methods.weight_prune(r, 80.)
    # r.set_mask(masks)
    # pruning.methods.prune_rate(r)

    # for name, child in r.named_children():
    #     if name == 'RCNN_base':
    #         pruning.methods.quantize_k_means(child, show_figures=True)




def main():
    parser = argparse.ArgumentParser(description='Prunes a network.')

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
