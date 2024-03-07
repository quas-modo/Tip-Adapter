#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 13:08
# @Author  : quasdo
# @Site    : 
# @File    : ood_dataset.py
# @Software: PyCharm

import os
import math
import random
from collections import defaultdict

import torchvision.transforms as transforms

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader

template = ['a photo of {}.']


class OutOfDomainDataset:
    """Get the out of domain dataset(needn't acknowledge the labels)
    """

    def __init__(self, dataset, root_path, log):
        self.dataset = dataset
        dataset_dir = os.path.join(root_path, self.dataset)
        dataset_dir = os.path.join(dataset_dir, 'images')
        self.dataset_dir = dataset_dir
        self.template = template
        self.log = log

        all = []
        image_extensions={'.jpg'}
        for dirpath, _, filenames in os.walk(self.dataset_dir):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in image_extensions:
                    impath = os.path.join(dirpath, filename)
                    item = Datum(
                        impath=impath,
                        label=-1,
                        classname="ood"
                    )
                    all.append(item)

        self._all = all

    @property
    def all(self):
        return self._all



