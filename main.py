from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mnist_classifier import MaskedMNist
from pruning.methods import weight_prune
from pruning.utils import to_var
from torchvision import datasets, transforms

def test(model, loader):

    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)
    for x, y in loader:
        x_var = to_var(x, volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples

    print('Test accuracy: {:.2f}% ({}/{})'.format(
        100.*acc,
        num_correct,
        num_samples,
        ))
    
    return acc

def main():

    # test_dataset = datasets.MNIST(root='../data/', train=False, download=True, 
    #     transform=transforms.ToTensor()
    # )
    
    # loader_test = torch.utils.data.DataLoader(test_dataset, 
    #     batch_size=64, shuffle=True
    # )

    # model = MaskedMNist().cuda()
    # model.load_state_dict(torch.load('/home/ahraz/Documents/thesis/pytorch_experiments/MNIST_demo_classifier/mnist_cnn.pt'))    
    # test(model, loader_test)

    # masks = weight_prune(model, 6.)
    # model.set_masks(masks)
    # test(model, loader_test)

    # torch.save(model.state_dict(), 'pruned_state_dict.pt')

if __name__ == '__main__':
    main()