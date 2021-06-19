# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
from torchvision.datasets.folder import ImageFolder, default_loader


def get_keep_index(labels, percent, num_classes, shuffle=False):
    labels = np.array(labels)
    keep_indexs = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        num_sample = len(idx)
        label_per_class = min(max(1, round(percent * num_sample)), num_sample)
        if shuffle:
            np.random.shuffle(idx)
        keep_indexs.extend(idx[:label_per_class])

    return keep_indexs


class ImageFolderWithPercent(ImageFolder):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, percent=1.0, shuffle=False):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)
        assert 0 <= percent <= 1
        if percent < 1:
            keep_indexs = get_keep_index(self.targets, percent, len(self.classes), shuffle)
            self.samples = [self.samples[i] for i in keep_indexs]
            self.targets = [self.targets[i] for i in keep_indexs]
            self.imgs = self.samples


class ImageFolderWithIndex(ImageFolder):

    def __init__(self, root, indexs=None, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)
        if indexs is not None:
            self.samples = [self.samples[i] for i in indexs]
            self.targets = [self.targets[i] for i in indexs]
            self.imgs = self.samples
