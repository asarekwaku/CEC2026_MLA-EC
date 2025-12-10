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
from dataset.basic_dataset import testDataSet
from dataset.basic_dataset import BaseClassificationDataset
from pathlib import Path



object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

def read_image_label(file):
    # print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
            # data.append([name, label])
            # print('%s  %d' % (name, label))
    return data


def read_object_labels(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = os.path.join(path_labels, object_categories[i] + '_' + set + '.txt')
        data = read_image_label(file)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w',newline='') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(20):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    # print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


def write_object_labels_csv_cat(file, images, object_categories):
    """
    将 images 写入 CSV 文件，保证 read_object_labels_csv 可以读取
    Args:
        file (str or Path): 输出 CSV 文件路径
        images (list of tuples): 每个元素是 (name, labels)，labels 是 torch.Tensor 或 list/numpy
        object_categories (list of str): 类别名称列表
    """
    print('[dataset] write file', file)

    # 打开文件写入

    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)

        # 写 header
        header = ['name'] + object_categories
        writer.writerow(header)

        # 写每一行
        for name, labels in images:
            # 如果是 Tensor，转换为 list
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
            writer.writerow([name] + labels)

    print(f'[dataset] saved {len(images)} items to {file}')


def find_images_classification(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    images = []
    file = os.path.join(path_labels, set + '.txt')
    with open(file, 'r') as f:
        for line in f:
            images.append(line)
    return images


class Voc2007Classification(BaseClassificationDataset):
    def __init__(self, root,  transform=None, phase='train', inp_name=None):
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        
        super().__init__(root, transform=transform, phase=phase)

        # download dataset
        #download_voc2007(self.root)

        # define path of csv file
        
        print("path",  root)
        path_csv = self.instanceDir(phase)
        

        
        # path_csv = os.path.join(self.root, 'files', 'VOC2007')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' +  self.phase + '.csv')
        print("path_csv:", file_csv)
        dir_path = os.path.dirname(os.path.abspath(__file__))
        # print("当前文件所在目录:", dir_path)
        inp_name= dir_path +"/files/voc2007/voc_glove_word2vec.pkl"
        self.adj_path = dir_path +"/files/voc2007/voc_adj.pkl"

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, 'VOC2007', self.phase)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        # with open(inp_name, 'rb') as f:
        #     self.inp = pickle.load(f)
        # self.inp_name = inp_name

        if inp_name is not None:
            with open(inp_name, 'rb') as f:
                self.inp = pickle.load(f)
            self.inp_name = inp_name
        else:
            self.inp = None
            self.inp_name = None

        print('[dataset] VOC 2007 classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.image_dir, path+'.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)


        # return (img, path, self.inp), target
        if self.inp is None:
            return (img, path), target
        else:
            return (img, path, self.inp), target    

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
    

    def adj_filepath(self):
        return self.adj_path
    
    def inp_filepath(self):
        return self.inp_name

    def name(self):
        return 'voc2007'

    def judgeOriginPhase(self, phase): 
        if phase in ['train', 'trainval', 'val','test']:
            return True
        else:
            return False

    def instanceDir(self, phase):
        if self.judgeOriginPhase(phase):
            # 如果 self.root 是字符串
            path_csv = Path(self.root) / 'files' / 'VOC2007'
            return str(path_csv)
        else:
            return self.instanceGenerateDir(phase)

    def outputFileter(self, filterIds, phase):
        if self.judgeOriginPhase(phase):
            raise "outputFileter failed: phase already in the dataset!"
        path_csv = self.instanceDir(phase)
        file_csv = os.path.join(path_csv,'classification_' +  phase + '.csv')
        new_images = [self.images[i] for i in filterIds]
        
        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # write csv file
            write_object_labels_csv_cat(file_csv, new_images, self.classes)

        return new_images

    def originImageDir(self):
        return self.path_images


    def imageName(self, index):
        path, target = self.images[index]
        return path + '.jpg'




class Voc2012Classification(BaseClassificationDataset):
    def __init__(self, root,transform=None, phase='train',  inp_name=None):
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        
        
        super().__init__(root,transform=transform, phase=phase)



        # download dataset
        #download_voc2007(self.root)

        # define path of csv file
        path_csv = self.instanceDir(phase)
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' +  self.phase + '.csv')




        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, 'VOC2012', self.phase)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        dir_path = os.path.dirname(os.path.abspath(__file__))
        inp_name= dir_path +"/files/voc2012/voc_glove_word2vec.pkl"
        self.adj_path = dir_path +"/files/voc2012/voc_adj.pkl"

        if inp_name is not None:
            with open(inp_name, 'rb') as f:
                self.inp = pickle.load(f)
            self.inp_name = inp_name
        else:
            self.inp = None
            self.inp_name = None

        # with open(inp_name, 'rb') as f:
        #     self.inp = pickle.load(f)
        # self.inp_name = inp_name

        print('[dataset] VOC 2012 classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.image_dir, path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)


     #   return (img, path, self.inp), target
    
        if self.inp is None:
            return (img, path), target
        else:
            return (img, path, self.inp), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
    
    def adj_filepath(self):
        return self.adj_path
    
    def inp_filepath(self):
        return self.inp_name

    def name(self):
        return 'voc2012'
    
    def outputFileter(self, filterIds, phase):
        if self.judgeOriginPhase(phase):
            raise "outputFileter failed: phase already in the dataset!"
        path_csv = self.instanceDir(phase)
        file_csv = os.path.join(path_csv, 'classification_' +  phase + '.csv')
        new_images = [self.images[i] for i in filterIds]
        
        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # write csv file
            write_object_labels_csv_cat(file_csv, new_images, self.classes)

        return new_images
  
    def judgeOriginPhase(self, phase): 
        if phase in ['train', 'trainval', 'val']:
            return True
        else:
            return False

    def instanceDir(self, phase):
        if self.judgeOriginPhase(phase):
            # 如果 self.root 是字符串
            path_csv = Path(self.root) / 'files' / 'VOC2012'
            return str(path_csv)
        else:
            return self.instanceGenerateDir(phase)



    def originImageDir(self):
        return self.path_images


    def imageName(self, index):
        path, target = self.images[index]
        return path + '.jpg'


    
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

#    voc2007_train = Voc2007Classification(root='./data/voc2007', transform=transform, phase='val')


    dataset = Voc2012Classification(root='./data/voc2012', transform=transform, phase='val')
    

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

    dataset = Voc2012Classification(root='./data/voc2012', transform=transform, phase='randselect')

    # # # 1. 获取数据集长度
    # # dataset_len = len(voc2007_train)  # __len__ 方法返回长度

    # # # 2. 随机选 100 个索引
    # # num_select = 100
    # # filterIds = random.sample(range(dataset_len), num_select)

    # # voc2007_train.outputFileter(filterIds, 'randselect')

    # # voc2007_train = Voc2007Classification(root='./data/voc2007', transform=transform, phase='randselect')


    # # # voc2007_val   = Voc2007Classification(root='./data/voc2007',transform=transform,  phase='val')

    # # testDataSet(voc2007_train)
    # # voc2007_train = Voc2012Classification(root='./data/voc2012',transform=transform, phase='train' )
    # # voc2012_val   = Voc2012Classification(root='./data/voc2012',transform=transform, phase='val')
    testDataSet(dataset)
  