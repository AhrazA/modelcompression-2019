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

from configurations import configurations

def main():
    parser = argparse.ArgumentParser(description="Quantizes a network.")

    parser.add_argument('--config', type=str, required=True, metavar="C",
                        help="Name of the configuration in configurations.py to run.")

    parser.add_argument('--pretrained-weights', required=True, type=str,
                        help="Path of the pretrained weights to load.")
    
    parser.add_argument('--save-model', action="store_true", default=False,
                        help='Save the quantized model.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()

    chosen_config = [x for x in configurations if x['name'] == args.config]

    if len(chosen_config) != 1:
        raise ValueError("Invalid configuration parameter.")
    
    if chosen_config[0]['type'] == 'classifier':
        wrapper = classifier_config(chosen_config[0], args)
    
    if chosen_config[0]['type'] == 'yolo':
        wrapper = yolo_config(chosen_config[0], args)

def classifier_config(config, args):
    model = config['model']()

    device = 'cuda:1' if not args.no_cuda else 'cpu'

    train_data = config['dataset'](
        './data', train=True, download=True, transform=transforms.Compose(config['transforms'])
    )

    test_data = config['dataset'](
        './data', train=False, download=True, transform=transforms.Compose(config['transforms'])
    )

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
    
    wrapper = Classifier(model, device, None, test_loader)

    model.load_state_dict(torch.load(args.pretrained_weights, map_location=device))

    wrapper.test(config["loss_fn"])

    print("Started quantizing")
    start_time = datetime.datetime.now()

    quantize_k_means(model, show_figures=True)

    prune_rate(model)
    
    end_time = datetime.datetime.now()
    print(f"Finished quantizing. Time taken: {end_time - start_time}")

    wrapper.test(config["loss_fn"])

    return wrapper

def yolo_config(config, args):
    config = [x for x in configurations if x['name'] == 'YOLOv3'][0]
    model = config['model'](config['config_path'])

    device = 'cuda:1' if not args.no_cuda else 'cpu'

    wrapper = YoloWrapper(device, model)
    lr0 = 0.001
    optimizer = config['optimizer'](filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=0.5)

    print("Loading dataloaders..")
    val_dataloader = LoadImagesAndLabels(config['datasets']['test'], batch_size=32, img_size=config['image_size'])

    model.to(device)

    print("Loading pretrained weights..")
    model.load_state_dict(torch.load(args.pretrained_weights, map_location=device))

    print("Pre-quantized percentage of zeros..")

    prune_rate(model)

    # with torch.no_grad():
    #     mAP, _, _ = wrapper.test(val_dataloader, img_size=config['image_size'], batch_size=32)
    #     print("Accuracy: ", mAP)

    print("Quantizing..")
    quantize_k_means(model)

    prune_rate(model)
    
    with torch.no_grad():
        mAP, _, _ = wrapper.test(val_dataloader, img_size=config['image_size'], batch_size=32)
        print("Accuracy: ", mAP)
    
    print("Post-quantize percentage of zeros..")

    prune_rate(model)

    if (args.save_model):
        torch.save(model.state_dict(), f'./models/{config["name"]}-quantized-{datetime.datetime.now().strftime("%Y%m%d%H%M")}.pt')
    
    return wrapper

if __name__ == '__main__':
    main()
