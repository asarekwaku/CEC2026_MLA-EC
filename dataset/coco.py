import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
from torchvision import datasets as tv_datasets
from dataset.basic_dataset import testDataSet
from dataset.basic_dataset import BaseClassificationDataset

def write_object_labels_json(anno, anno_list):
    if not os.path.exists(anno):
        json.dump(anno_list, open(anno, 'w'))


def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO2014(BaseClassificationDataset):
    def __init__(self, root, transform=None, phase='train'):
        self.root = root
        super().__init__(root,transform=transform, phase=phase)
        self.img_list = []
        #download_coco2014(root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)
        
        dir_path = os.path.dirname(os.path.abspath(__file__))
        # print("当前文件所在目录:", dir_path)
        inp_name= dir_path +"/files/coco/coco_glove_word2vec.pkl"
        self.adj_path = dir_path +"/files/coco/coco_adj.pkl"
        if inp_name is not None:
            with open(inp_name, 'rb') as f:
                self.inp = pickle.load(f)
            self.inp_name = inp_name
        else:
            self.inp = None
            self.inp_name = None

    def get_anno(self):
        pathdir= self.instanceDir(self.phase)
        list_path = os.path.join(pathdir, '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root,'adv_filter_data','coco', 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
        # if self.judgeOriginPhase(self.phase):

        # else:
        #     img = Image.open(os.path.join(self.root, 'val2014', filename)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1

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
        return 'coco'
    

    def outputFileter(self, filterIds, phase):
        if self.judgeOriginPhase(phase):
            raise "outputFileter failed: phase already in the dataset!"
        pathdir= self.instanceDir(phase)
        if not os.path.exists(pathdir):  # create dir if necessary
            os.makedirs(pathdir)

        new_imageList = [self.img_list[i] for i in filterIds]

        save_path = os.path.join(pathdir, '{}_anno.json'.format(phase))
        with open(save_path, 'w') as f:
            json.dump(new_imageList, f, indent=4)



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
        return os.path.join(self.root, '{}2014'.format(self.phase))


    def imageName(self, index):
        item = self.img_list[index]
        filename = item['file_name']
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


    # voc2007_train = COCO2014(root='./MLAE_competition/data/coco/data', phase='train', transform=transform)
    # voc2007_val   = COCO2014(root='./MLAE_competition/data/coco/data', phase='val', transform=transform)

    # voc2007_train = COCO2014(root='./data/coco/data', phase='train', transform=transform)
    dataset   = COCO2014(root='./data/coco/data',  transform=transform, phase='val')

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
        

    dataset = COCO2014(root='./data/coco/data', transform=transform, phase='randselect')


    testDataSet(dataset)