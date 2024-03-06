import os
import random
from scipy.io import loadmat
from collections import defaultdict

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase, read_json


template = ['a photo of a {}, a type of flower.']


class OxfordFlowers(DatasetBase):

    dataset_dir = 'oxford_flowers'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'jpg')
        self.label_file = os.path.join(self.dataset_dir, 'imagelabels.mat') # 标签
        self.lab2cname_file = os.path.join(self.dataset_dir, 'cat_to_name.json') # 标签到类名的对应
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_OxfordFlowers.json') # 划分训练集和验证集
        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir) # 分别读取train val 和test的数据集
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        
        super().__init__(train_x=train, val=val, test=test)


    def read_split(self, filepath, path_prefix, data_section=0):
        """
        从oxford_pets.split_path更改
        data_section: 选择训练的数据范围, 0表示不分割， 1表示前半部分，2表示后半部分
        """
        def _convert(items, data_section=0):
            out = []
            for impath, label, classname in items:
                if data_section == 1:
                    if int(label) > (num_labels // 2):
                        continue
                elif data_section == 2:
                    if int(label) < (num_labels // 2):
                        continue
                impath = os.path.join(path_prefix, impath)
                item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=classname
                )
                out.append(item)
            return out

        def _get_num_labels(items):
            label_set = set()
            for impath, label, classname in items:
                label_set.add(label)
            return max(label_set) + 1


        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        num_labels = _get_num_labels(split['train'])
        print(f'Num_label Total:  {num_labels}')
        train = _convert(split['train'], data_section=data_section)
        val = _convert(split['val'], data_section=data_section)
        test = _convert(split['test'], data_section=data_section)

        # select open-set classes and samples
        else_data_section = 0
        if data_section == 1:
            else_data_section = 2
        elif data_section == 2:
            else_data_section = 1
        open_set = _convert(split['test'], data_section=else_data_section)

        return train, val, test, open_set
    
    def read_data(self):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)['labels'][0]
        for i, label in enumerate(label_file):
            imname = f'image_{str(i + 1).zfill(5)}.jpg'
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)
        
        print('Splitting data into 50% train, 20% val, and 30% test')

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(
                    impath=im,
                    label=y-1, # convert to 0-based label
                    classname=c
                )
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train:n_train+n_val], label, cname))
            test.extend(_collate(impaths[n_train+n_val:], label, cname))
        
        return train, val, test