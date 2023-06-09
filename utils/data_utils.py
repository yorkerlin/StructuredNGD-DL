import torch
import os
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_tiny():
    data_path = 'data/tiny-imagenet-200/'

    def _train_dataset(path):
        normalize = transforms.Normalize(
            mean=[0.4802, 0.4481, 0.3975],
            std=[0.2302, 0.2265, 0.2262],
        )

        train_dir = os.path.join(path, 'train')
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                normalize,
            ]))
        return train_dataset


    def _test_dataset(path):
        normalize = transforms.Normalize(
            mean=[0.4802, 0.4481, 0.3975],
            std=[0.2302, 0.2265, 0.2262],
        )
        val_dir = os.path.join(path, 'test')
        dataset = datasets.ImageFolder(val_dir, transforms.Compose(
            [
                transforms.ToTensor(),
                normalize, ]))
        return dataset

    return _train_dataset(data_path), _test_dataset(data_path)


def get_transforms(dataset):
    transform_train = None
    transform_test = None
    if dataset== 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    if dataset== 'stl10':
        transform_train = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    if dataset.find('tinyimagenet')>=0 or  dataset.find('imagenet100')>=0:
        return None, None

    assert transform_test is not None and transform_train is not None, 'Error, no dataset %s' % dataset
    return transform_train, transform_test


def get_dataloader(dataset, train_batch_size, test_batch_size, num_workers=2, root='./data'):
    transform_train, transform_test = get_transforms(dataset)
    trainset, testset = None, None
    if dataset.find('cifar100')>=0:
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    elif dataset.find('cifar10')>=0:
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    elif dataset.find('stl10')>=0:
        trainset = torchvision.datasets.STL10(root=root, split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.STL10(root=root, split='test', download=True, transform=transform_test)
    elif dataset.find('tinyimagenet')>=0:
        trainset, testset = get_tiny()
    elif dataset.find('imagenet100')>=0:
        trainset, testset = get_imagenet100()



    assert trainset is not None and testset is not None, 'Error, no dataset %s' % dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers)


    print(dataset,  len(trainloader.dataset), len(testloader.dataset) ) 
    return trainloader, testloader


def get_imagenet100():
    data_path = 'data/imagenet100/'

    def _train_dataset(path):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        train_dir = os.path.join(path, 'train')
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        return train_dataset


    def _test_dataset(path):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        val_dir = os.path.join(path, 'test')
        dataset = datasets.ImageFolder(val_dir, transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize, ]))
        return dataset

    return _train_dataset(data_path), _test_dataset(data_path)

