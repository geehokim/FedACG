from utils.registry import Registry
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Normalize, CenterCrop
import yaml
import torch

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
"""
DATASET_REGISTRY.register(CIFAR10)
DATASET_REGISTRY.register(CIFAR100)

__all__ = ['build_dataset', 'build_datasets']


def get_transform(args, train, config):
    if 'leaf_femnist' in args.dataset.name:
        transform = transforms.Compose([
            transforms.Resize(size=(28, 28)),
            ToTensor()])
    elif 'leaf_celeba' in args.dataset.name:
        transform = transforms.Compose([
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    else:
        color_jitter = transforms.ColorJitter(0.4 * 1, 0.4 * 1, 0.4 * 1, 0.1 * 1)
        normalize = transforms.Normalize(config['mean'],
                                         config['std'])
        imsize = config['imsize']
        if train:
            transform = transforms.Compose(
                [transforms.RandomRotation(10),
                 transforms.RandomCrop(imsize, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 normalize
                 ])
        else:
            transform = transforms.Compose(
                [transforms.CenterCrop(imsize),
                 transforms.ToTensor(),
                 normalize])

    return transform


def build_dataset(args, train=True):
    if args.verbose and train == True:
        print(DATASET_REGISTRY)

    download = args.dataset.download if args.dataset.get('download') else False

    with open('datasets/configs.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)[args.dataset.name]
    transform = get_transform(args, train, dataset_config)
    dataset = DATASET_REGISTRY.get(args.dataset.name)(root=args.dataset.path, download=download, train=train, transform=transform) if len(args.dataset.path) > 0 else None

    return dataset


def build_datasets(args):
    train_dataset = build_dataset(args, train=True)
    test_dataset = build_dataset(args, train=False)
    
    datasets = {
        "train": train_dataset,
        "test": test_dataset,
    }

    return datasets