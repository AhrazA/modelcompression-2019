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
    config = [x for x in configurations if x['name'] == 'MNistClassifier'][0]

    model = config['model']()

    device = 'cuda:0'

    train_data = test_data = config['dataset'](
        './data', train=True, download=True, transform=transforms.Compose(config['transforms'])
    )

    test_data = config['dataset'](
        './data', train=False, download=True, transform=transforms.Compose(config['transforms'])
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)
    optimizer = config['optimizer'](model.parameters(), lr=0.01, momentum=0.5)
    
    wrapper = Classifier(model, device, train_loader, test_loader)

    model.load_state_dict(torch.load('./models/mnist_classifier.pt'))

    print("Started quantizing")
    start_time = datetime.datetime.now()

    quantize_k_means(model)
    
    end_time = datetime.datetime.now()
    print(f"Finished quantizing. Time taken: {end_time - start_time}")

    wrapper.test(config['loss_fn'])

    optimizer = config['optimizer'](model.parameters(), lr=0.01, momentum=0.5)
    
    wrapper.train(10, optimizer, 1, config['loss_fn'])

    # config = [x for x in configurations if x['name'] == 'YOLOv3'][0]
    # model = config['model'](config['config_path'])
    # device = 'cuda:1'
    # wrapper = YoloWrapper(device, model)
    # lr0 = 0.001
    # optimizer = config['optimizer'](filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=0.5)

    # print("Loading dataloaders..")
    # val_dataloader = LoadImagesAndLabels(config['datasets']['test'], batch_size=32, img_size=config['image_size'])

    # model.to(device)

    # print("Loading pretrained weights..")
    # model.load_state_dict(torch.load('./models/yolov3.pt')['model'])

    # print("Quantizing..")
    # quantize_k_means(model)

    # with torch.no_grad():
    #     wrapper.test(val_dataloader, img_size=config['image_size'], batch_size=32)