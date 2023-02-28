import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, TrivialAugmentWide, InterpolationMode, RandomHorizontalFlip, \
    RandomCrop, RandomErasing

from lib.Experiment import IExperiment
from lib.utils.dist_utils import is_distributed


def get_default_validation_transforms():
    return [ToTensor(), Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])]


def get_default_train_transforms():
    return [
        TrivialAugmentWide(interpolation=InterpolationMode(value="bilinear")),
        RandomHorizontalFlip(),
        RandomCrop(size=32, padding=4),
        ToTensor(),
        RandomErasing(p=0.1),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ]


def patch(exp: IExperiment):
    transform_train = get_default_train_transforms()
    transform_validation = get_default_validation_transforms()

    transform_train = transforms.Compose(transform_train)
    transform_validation = transforms.Compose(transform_validation)

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    validation_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                  transform=transform_validation)

    local_batch_size = 16

    print("Local batch size:", local_batch_size)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                    shuffle=True) if is_distributed() else None
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_set,
                                                                         shuffle=False) if is_distributed() else None

    exp.train_loader = torch.utils.data.DataLoader(train_set, batch_size=local_batch_size, num_workers=4,
                                                   sampler=train_sampler,
                                                   pin_memory=True)

    exp.val_loader = torch.utils.data.DataLoader(validation_set, batch_size=local_batch_size, num_workers=4,
                                                 sampler=validation_sampler,
                                                 pin_memory=True)
