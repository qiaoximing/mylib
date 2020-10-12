import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

NORMALIZE = {
    'mnist': ((0.1307,), (0.3081,)),
    'fmnist': ((0.1307,), (0.3081,)),
    'cifar': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}
def inverse(means, stds):
    # y = (x - mean) / std
    # x = y * std + mean
    # x = (y + (mean / std)) * std
    # x = (y - (-mean / std)) / (1 / std)
    return tuple(-mean / std for (mean, std) in zip(means, stds)), tuple(1 / std for std in stds)

DATA_SHAPE = {
    'mnist': (1, 28, 28),
    'fmnist': (1, 28, 28),
    'cifar': (3, 32, 32),
    'imagenet': (3, 224, 224)
}

def get_dataset(data_name, data_dir=None):
    if data_name == 'mnist':
        data_dir = 'data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*NORMALIZE['mnist'])])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    if data_name == 'fmnist':
        data_dir = 'data/fmnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*NORMALIZE['fmnist'])])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    if data_name == 'cifar':
        data_dir = 'data/cifar/'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*NORMALIZE['cifar'])])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*NORMALIZE['cifar'])])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=transform_test)

    if data_name == 'imagenet':
        data_dir = '/home/public/ImageNet'
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(*NORMALIZE['imagenet'])

        train_dataset = datasets.ImageFolder(traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        test_dataset = datasets.ImageFolder(valdir, 
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    return train_dataset, test_dataset

class Data():
    def __init__(self, data_name):
        super().__init__()
        self.data_name = data_name
        self.trainset, self.testset = get_dataset(data_name)
        self.num_classes = len(self.testset.classes)
        self.data_shape = DATA_SHAPE[data_name]
        self.normalizer = transforms.Normalize(*NORMALIZE[self.data_name])
        self.denormalizer = transforms.Normalize(*inverse(*NORMALIZE[self.data_name]))

    def trainloader(self, args={}):
        default_args = {
            'batch_size': 128,
            'shuffle': True,
            'num_workers': 4}
        default_args.update(args); args = default_args
        return DataLoader(self.trainset, **args)

    def testloader(self, args={}):
        default_args = {
            'batch_size': 128,
            'shuffle': True,
            'num_workers': 4}
        default_args.update(args); args = default_args
        return DataLoader(self.testset, **args)

    def plot(self, data, ncol=8):
        # n = len(data)
        # fig, axs = plt.subplots((n + ncol - 1) // ncol, ncol, figsize=(10,10))
        # for i in range(n):
        #     img = self.denormalizer(data[i])
        #     img = transforms.ToPILImage()(img)
        #     if n <= ncol:
        #         axs[i].imshow(img)
        #     else:
        #         axs[i // ncol][i % ncol].imshow(img)
        # fig.show()
        imgs = []
        for img in data:
            x = self.denormalizer(img)
            imgs.append(x)
        imgs = make_grid(imgs, ncol)
        plt.figure(figsize=(10,10))
        plt.imshow(transforms.ToPILImage()(imgs))
        plt.show()