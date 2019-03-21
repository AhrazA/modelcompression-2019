from cifar_classifier import MaskedCifar
from mnist_classifier import MaskedMNist
from resnet import MaskedResNet18
from yolov3 import MaskedDarknet, YoloWrapper
from classifier import Classifier
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

configurations = [
    {
        'name': 'FCCifar10Classifier',
        'type': 'classifier',
        'model': MaskedCifar,
        'wrapper': Classifier,
        'dataset': datasets.CIFAR10,
        'transforms': 
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ],
        'loss_fn': F.cross_entropy,
        'optimizer': optim.SGD
    },
    {
        'name': 'MNistClassifier',
        'type': 'classifier',
        'model': MaskedMNist,
        'wrapper': Classifier,
        'dataset': datasets.MNIST,
        'transforms':
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ],
        'loss_fn': F.nll_loss,
        'optimizer': optim.SGD
    },
    {
        'name': 'ResNet18CifarClassifier',
        'type': 'classifier',
        'model': MaskedResNet18,
        'wrapper': Classifier,
        'dataset': datasets.CIFAR10,
        'transforms':
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ],
        'loss_fn': F.cross_entropy,
        'optimizer': optim.SGD
    },
    {
        'name': 'YOLOv3',
        'type': 'yolo',
        'model': MaskedDarknet,
        'wrapper': YoloWrapper,
        'datasets': {
            'train': './data/yolo/train.txt',
            'test': './data/yolo/test.txt',
            'val': './data/yolo/train.txt',
        },
        'optimizer': optim.SGD
    }
]