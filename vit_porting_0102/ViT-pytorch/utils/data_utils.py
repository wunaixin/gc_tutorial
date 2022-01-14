import logging

import poptorch

from torchvision import transforms, datasets


logger = logging.getLogger(__name__)


def get_loader(args, opts=None):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None

    train_loader = poptorch.DataLoader(
        options=opts,
        shuffle=True,
        dataset=trainset,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True)
    test_loader = poptorch.DataLoader(
        options=opts,
        shuffle=False,
        dataset=testset,
        batch_size=args.eval_batch_size,
        num_workers=4,
        pin_memory=True) if testset is not None else None

    return train_loader, test_loader
