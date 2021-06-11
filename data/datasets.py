# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from PIL import Image
import os
import os.path
import pandas as pd
import numpy as np
import random
from torchvision.datasets.vision import VisionDataset


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class OpenImagesDataset(VisionDataset):

    def __init__(self, image_root, annotation_file, transform=None, target_transform=None,
                 loader=default_loader):
        super(OpenImagesDataset, self).__init__(image_root, transform=transform,
                                                              target_transform=target_transform)
        line = pd.read_csv(annotation_file)
        label_names = line['LabelName'].tolist()
        self.classes = sorted(list(set(label_names)))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        image_paths = line['ImageID'].tolist()
        samples = []
        for (img_id, label_name) in zip(image_paths, label_names):
            img_path = os.path.join(self.root, img_id)
            item = (img_path, self.class_to_idx[label_name])
            samples.append(item)
        self.samples = samples
        self.loader = loader
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


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


class OpenImagesDatasetWithPercent(OpenImagesDataset):

    def __init__(self, image_root, annotation_file, transform=None, target_transform=None,
                 loader=default_loader, percent=1.0, shuffle=False):
        super().__init__(image_root, annotation_file,
                         transform=transform,
                         target_transform=target_transform,
                         loader=loader)
        assert 0 <= percent <= 1
        if percent < 1:
            keep_indexs = get_keep_index(self.targets, percent, len(self.classes), shuffle)
            self.samples = [self.samples[i] for i in keep_indexs]
            self.targets = [self.targets[i] for i in keep_indexs]
            self.imgs = self.samples


class OpenImagesDatasetFewShot(VisionDataset):

    def __init__(self, image_root, annotation_file, transform=None, target_transform=None,
                 loader=default_loader):
        super(OpenImagesDatasetFewShot, self).__init__(image_root, transform=transform,
                                                       target_transform=target_transform)
        line = pd.read_csv(annotation_file)
        label_names = line['LabelName'].tolist()
        self.classes = sorted(list(set(label_names)))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        image_paths = line['ImageID'].tolist()
        samples = []
        for (img_id, label_name) in zip(image_paths, label_names):
            img_path = os.path.join(self.root, img_id)
            item = (img_path, self.class_to_idx[label_name])
            samples.append(item)
        self.samples = samples
        self.loader = loader
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples
        self.low_shot = False

    def convert_low_shot(self, k):

        cls2sample = {c: [] for c in range(len(self.classes))}

        for sample, cls in zip(self.samples, self.targets):
            cls2sample[cls].append(sample)

        self.samples_lowshot = []
        for cls, samplelist in cls2sample.items():
            random.shuffle(samplelist)
            self.samples_lowshot += samplelist[:k]
        self.low_shot = True

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.low_shot:
            path, target = self.samples_lowshot[index]
        else:
            path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.low_shot:
            return len(self.samples_lowshot)
        else:
            return len(self.samples)


class OpenImagesDatasetWithIndex(OpenImagesDataset):

    def __init__(self, image_root, annotation_file, indexs, transform=None, target_transform=None,
                 loader=default_loader):
        super().__init__(image_root, annotation_file,
                         transform=transform,
                         target_transform=target_transform,
                         loader=loader)
        if indexs is not None:
            self.samples = [self.samples[i] for i in indexs]
            self.targets = [self.targets[i] for i in indexs]
            self.imgs = self.samples
