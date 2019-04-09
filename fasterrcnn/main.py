import sys
sys.path.append('/home/ahraz/Documents/thesis/pytorch_experiments/weights_pruning_demo')

from resnet import resnet


if __name__ == '__main__':
    r = resnet(('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    
    r.create_architecture()