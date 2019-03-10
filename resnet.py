import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.optim as optim

from pruning.masked_conv_2d import MaskedConv2d
from pruning.masked_linear import MaskedLinear
from pruning.methods import weight_prune, prune_rate
from pruning.masked_sequential import MaskedSequential
from pruning.masked_conv_2d import conv1x1, conv3x3
from classifier_utils import setup_default_args
from classifier import Classifier
from torchvision import datasets, transforms
import torch.optim as optim

import numpy as np

class MaskedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MaskedBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    def set_mask(self, masks):
        self.conv1.set_mask(masks[0])
        self.conv2.set_mask(masks[1])

class MaskedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MaskedBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    def set_mask(self, masks):
        self.conv1.set_mask(masks[0])
        self.conv2.set_mask(masks[1])
        self.conv3.set_mask(masks[2])

class MaskedResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(MaskedResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = MaskedConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = MaskedLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, MaskedBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, MaskedBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = MaskedSequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return MaskedSequential(*layers)

    def set_mask(self, masks):
        self.conv1.set_mask(masks[0])
        self.layer1.set_mask(masks[1])
        self.layer2.set_mask(masks[2])
        self.layer3.set_mask(masks[3])
        self.layer4.set_mask(masks[4])
        self.fc.set_mask(masks[5])

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class MaskedResNet18(MaskedResNet):
    def __init__(self, **args):
        super(MaskedResNet18, self).__init__(MaskedBasicBlock, [2, 2, 2, 2], **args)

class MaskedResNet34(MaskedResNet):
    def __init__(self, **args):
        super(MaskedResNet34, self).__init__(MaskedBasicBlock, [3, 4, 6, 3], **args)

class MaskedResNet50(MaskedResNet):
    def __init__(self, **args):
        super(MaskedResNet50, self).__init__(MaskedBottleneck, [3, 4, 6, 3], **args)

class MaskedResNet101(MaskedResNet):
    def __init__(self, **args):
        super(MaskedResNet101, self).__init__(MaskedBottleneck, [3, 4, 23, 3], **args)

class MaskedResNet152(MaskedResNet):
    def __init__(self, **args):
        super(MaskedResNet152, self).__init__(MaskedBottleneck, [3, 8, 36, 3], **args)

configurations = {
    'resnet18': {
        'type': MaskedResNet18,
        'model_url': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'model_args': {}
    },
    'resnet34': {
        'type': MaskedResNet34,
        'model_url': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'model_args': {}
    },
    'resnet50': {
        'type': MaskedResNet50,
        'model_url': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'model_args': {}
    },
    'resnet101': {
        'type': MaskedResNet101,
        'model_url': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'model_args': {}
    },
    'resnet152': {
        'type': MaskedResNet152,
        'model_url': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'model_args': {}
    }
}

def get_resnet(resnet_type_name, config, pretrained=False, **kwargs):
    config = configurations[resnet_type_name]
    model = config['type'](**config['model_args'], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(config['model_url']))
    return model

def get_all_weights_for_nested_model(model):
    weights = []

    if len(list(model.children())) != 0:
        for l in model.children():
            weights += get_all_weights_for_nested_model(l)
    else:
        for p in model.parameters():
            if len(p.data.size()) != 1: # Avoid bias parameters
                weights += list(p.cpu().data.abs().numpy().flatten())

    return weights

def mask_model(model, pruning_perc):
    masks = weight_prune(model, pruning_perc)
    model.set_mask(masks)
    prune_rate(model)
    return model

def main():
    parser = argparse.ArgumentParser(description='PyTorch ResNet Example')
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
    
    for c in configurations:
        model = get_resnet(c, configurations[c], pretrained=True)
        classifier = Classifier(model, 'cuda', train_loader, test_loader)
        optimizer = optim.SGD(classifier.model.parameters(), lr=args.lr, momentum=args.momentum)

        for epoch in range(1, args.epochs + 1):
            classifier.train(args.log_interval, optimizer, epoch, F.cross_entropy)
            classifier.test(F.cross_entropy)
        
        if (args.save_model):
            torch.save(classifier.model.state_dict(),f"models/{c}.pt")

if __name__ == '__main__':
    main()