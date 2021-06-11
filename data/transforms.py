# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import random
from PIL import ImageOps, ImageFilter
from torchvision import transforms

from .randaugment import RandAugment


class Solarize(object):
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        return ImageOps.solarize(img, self.threshold)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_augmentations(aug_type):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    default_train_augs = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    default_val_augs = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    appendix_augs = [
        transforms.ToTensor(),
        normalize,
    ]
    if aug_type == 'DefaultTrain':
        augs = default_train_augs + appendix_augs
    elif aug_type == 'DefaultVal':
        augs = default_val_augs + appendix_augs
    elif aug_type == 'RandAugment':
        augs = default_train_augs + [RandAugment(n=2, m=10)] + appendix_augs
    elif aug_type == 'MoCoV1':
        augs = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip()
        ] + appendix_augs
    elif aug_type == 'MoCoV2':
        augs = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ] + appendix_augs
    else:
        raise NotImplementedError('augmentation type not found: {}'.format(aug_type))

    return augs


def get_transforms(aug_type):
    augs = get_augmentations(aug_type)
    return transforms.Compose(augs)


def get_byol_tranforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.),
        transforms.RandomApply([Solarize()], p=0.),
        transforms.ToTensor(),
        normalize
    ]
    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([Solarize()], p=0.2),
        transforms.ToTensor(),
        normalize
    ]
    transform1 = transforms.Compose(augmentation1)
    transform2 = transforms.Compose(augmentation2)
    return transform1, transform2


class TwoCropsTransform:
    """Take two random crops of one image."""

    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        out1 = self.transform1(x)
        out2 = self.transform2(x)
        return out1, out2

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        names = ['transform1', 'transform2']
        for idx, t in enumerate([self.transform1, self.transform2]):
            format_string += '\n'
            t_string = '{0}={1}'.format(names[idx], t)
            t_string_split = t_string.split('\n')
            t_string_split = ['    ' + tstr for tstr in t_string_split]
            t_string = '\n'.join(t_string_split)
            format_string += '{0}'.format(t_string)
        format_string += '\n)'
        return format_string
