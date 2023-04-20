#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:35:03 2023

@author: sotarosuzuki
"""

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

class ImageClassificationDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = os.listdir(root)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        images = []
        for target_class in os.listdir(self.root):
            target_dir = os.path.join(self.root, target_class)
            if not os.path.isdir(target_dir):
                continue
            for filename in os.listdir(target_dir):
                path = os.path.join(target_dir, filename)
                item = (path, self.class_to_idx[target_class])
                images.append(item)
        return images

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)
    
date = ImageClassificationDataset(root="/Users/sotarosuzuki/Downloads/cifar10/cifar10/train")
image, label = data
