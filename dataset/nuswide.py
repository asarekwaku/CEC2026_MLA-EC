import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import ast

from dataset.basic_dataset import testDataSet
from dataset.basic_dataset import BaseClassificationDataset
from pathlib import Path

tags = ['airport', 'animal', 'beach', 'bear', 'birds', 'boats', 'book', 'bridge', 'buildings', 'cars', 'castle', 'cat',
        'cityscape', 'clouds', 'computer', 'coral', 'cow', 'dancing', 'dog', 'earthquake', 'elk', 'fire', 'fish',
        'flags', 'flowers', 'food', 'fox', 'frost', 'garden', 'glacier', 'grass', 'harbor', 'horses', 'house', 'lake',
        'leaf', 'map', 'military', 'moon', 'mountain', 'nighttime', 'ocean', 'person', 'plane', 'plants', 'police',
        'protest', 'railroad', 'rainbow', 'reflection', 'road', 'rocks', 'running', 'sand', 'sign', 'sky', 'snow',
        'soccer', 'sports', 'statue', 'street', 'sun', 'sunset', 'surf', 'swimmers', 'tattoo', 'temple', 'tiger',
        'tower', 'town', 'toy', 'train', 'tree', 'valley', 'vehicle', 'water', 'waterfall', 'wedding', 'whales',
        'window', 'zebra']


def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class NusWide(BaseClassificationDataset):
    def __init__(self, root, transform=None, phase='train', inp_name=None):
        self.root = root
        super().__init__(root,transform=transform, phase=phase)

        self.num_classes = 81
        self.get_anno()
        self.tags = tags
        dir_path = os.path.dirname(os.path.abspath(__file__))
        # print("当前文件所在目录:", dir_path)
        inp_name= dir_path +"/files/nus/glove_word2vec.pkl"
        self.adj_path = dir_path +"/files/nus/nuswide_adj.pkl"

        if inp_name is not None:
            with open(inp_name, 'rb') as f:
                self.inp = pickle.load(f)
            self.inp_name = inp_name
        else:
            self.inp = None
            self.inp_name = None

    def get_anno(self):
        self.img_name_list = []
        self.tag_list = []
        img_list_path = os.path.join(self.instanceDir(self.phase), 'nus_wide_data_{}.csv'.format(self.phase))
        with open(img_list_path, 'r') as f:
            reader = csv.reader(f)
            rownum = 0
            for row in reader:
                if rownum == 0:
                    pass
                else:
                    self.img_name_list.append(row[0].split('/')[1])
                    tag_names = ast.literal_eval(row[1])
                    tag = [-1 for i in range(self.num_classes)]
                    for tag_name in tag_names:
                        tag[tags.index(tag_name)] = 1
                    self.tag_list.append(tag)
                rownum += 1
    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index):
        filename = self.img_name_list[index]
        tag = self.tag_list[index]
        img = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.asarray(tag)
        target[target==0] = -1

        if self.inp is None:
            return (img, filename), target
        else:
            return (img, filename, self.inp), target

    def get_number_classes(self):
        return self.num_classes

    def adj_filepath(self):
        return self.adj_path
    
    def inp_filepath(self):
        return self.inp_name
    
    def name(self):
        return 'nuswide'
    
    def outputFileter(self, filterIds, phase):
        if self.judgeOriginPhase(phase):
            raise "outputFileter failed: phase already in the dataset!"

        new_image = [self.img_name_list[i] for i in filterIds]
        new_tag = [self.tag_list[i] for i in filterIds]
        pathdir= self.instanceDir(phase)
        if not os.path.exists(pathdir):  # create dir if necessary
            os.makedirs(pathdir)
        img_list_path = os.path.join(pathdir, 'nus_wide_data_{}.csv'.format(phase))
        with open(img_list_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filepath', 'label', 'split_name'])
            for img, tag in zip(new_image, new_tag):
                tags = []
                for i, t in enumerate(tag):
                    if t == 1:
                        tags.append(self.tags[i])
                row_info = ['images/' + img, str(tags), phase]
                writer.writerow(row_info)
        

    def instanceDir(self, phase):
        if self.judgeOriginPhase(phase):
            return self.root
        else:
            return self.instanceGenerateDir(phase)

    def judgeOriginPhase(self, phase): 
        if phase in ['train', 'val']:
            return True
        else:
            return False

    def originImageDir(self):
        return os.path.join(self.root, 'images')


    def imageName(self, index):
        filename = self.img_name_list[index]
        return filename



import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import shutil


if __name__ == '__main__':



    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # 调整大小
        transforms.ToTensor(),           # 转为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                            std=[0.229, 0.224, 0.225])   # ImageNet 方差
    ])

    dataset = NusWide(root='./data/NUSWIDE', transform=transform, phase='val')
    

    dataset_len = len(dataset)  # __len__ 方法返回长度

    
        # # # 2. 随机选 100 个索引
    num_select = 100
    filterIds = random.sample(range(dataset_len), num_select)

    dataset.outputFileter(filterIds, 'randselect')
    originDir = dataset.originImageDir()
    saveDir   = dataset.imageDir('randselect')

    os.makedirs(saveDir, exist_ok=True)

    for it in filterIds:
        src = os.path.join(originDir, dataset.imageName(it))
        dst = os.path.join(saveDir, dataset.imageName(it))
        shutil.copy(src, dst)
        
    dataset = NusWide(root='./data/NUSWIDE', transform=transform, phase='randselect')
      
        
    # voc2007_val   = NusWide(root='./data/NUSWIDE', phase='val', transform=transform)
    # num_select = 100
    # dataset_len = len(voc2007_train)  # __len__ 方法返回长度

    # filterIds = random.sample(range(dataset_len), num_select)

    # voc2007_train.outputFileter(filterIds, 'randselect')
    # voc2007_train = NusWide(root='./data/NUSWIDE', transform=transform, phase='randselect')

    testDataSet(dataset)