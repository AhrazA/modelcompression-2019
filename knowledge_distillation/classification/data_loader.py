import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar(num_classes = 100, data_dir = './data', batch_size = 128, crop = False):

    normalization = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    simple_transform = transforms.Compose([transforms.ToTensor(), normalization])

    if crop is True:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization
        ])
    else:
        train_transform = simple_transform

    if num_classes == 100:
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=train_transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=train_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False)

    return trainloader, testloader


# if __name__ == "__main__":
#     print("CIFAR10")
#     print(get_cifar(10))
#     print("---"*20)
# 	print("---"*20)
# 	print("CIFAR100")
# 	print(get_cifar(100))




