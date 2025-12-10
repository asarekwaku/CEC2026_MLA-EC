import os
from torch.utils import data
from abc import ABC, abstractmethod
from pathlib import Path


class BaseClassificationDataset(data.Dataset, ABC):
    """A base class for multi-label classification datasets."""

    def __init__(self, datadir, transform=None, phase='train'):
        """Initialize the dataset class. This method must be implemented by the subclass."""
        super().__init__()
        # # Any common initialization can be done here
        self.transform = transform
        self.phase = phase
        self.datadir= datadir
        
        if self.judgeOriginPhase(phase):
            self.image_dir = self.originImageDir()
        else:
            self.image_dir = self.imageDir(phase)



    @abstractmethod
    def get_number_classes(self):
        """Return the number of classes in the dataset. This method must be implemented by the subclass."""
        pass
    
    @abstractmethod
    def adj_filepath(self):
        pass
    
    @abstractmethod
    def inp_filepath(self):
        pass

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def outputFileter(self, filterIds, phase):
        pass

    @abstractmethod
    def instanceDir(self, phase):
        pass

    def instanceGenerateDir(self, phase):
        #     # 当前文件所在目录
        # dir_path = Path(__file__).resolve().parent
        # path = dir_path.parent
        # # 拼接 CSV 文件路径
        # path_csv = path /'project' / 'adv_filter_data' / self.name()/ phase
        
        path_csv= os.path.join(self.datadir, 'adv_filter_data', self.name(), phase   )
        return str(path_csv)

    @abstractmethod
    def originImageDir(self):
        pass

    @abstractmethod
    def imageName(self, index):
        pass
    
    def imageDir(self,  phase):
        path_csv = self.instanceGenerateDir(phase)
        return os.path.join(path_csv, 'images')

    @abstractmethod
    def judgeOriginPhase(self, phase): 
        pass



import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def testDataSet(dataset: BaseClassificationDataset):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # 调整大小
        transforms.ToTensor(),           # 转为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                            std=[0.229, 0.224, 0.225])   # ImageNet 方差
    ])

    # voc2007_train = COCO2014(root='./data/coco/data', phase='train', transform=transform)
    # voc2007_val   = COCO2014(root='./data/coco/data', phase='val', transform=transform)

    train_loader_2007 = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


    for (imgs, paths, inp), targets in train_loader_2007:
    # imgs: Tensor [batch, 3, H, W]
    # paths: 图像文件名列表
    # targets: label
        print(imgs.shape, targets.shape)
        break




    # 测试 __len__()
    print("数据集大小:", len(dataset))

    # 测试 get_number_classes()
    print("类别数量:", dataset.get_number_classes())

    # 测试 __getitem__()
    sample_index = 0  # 测试第0个样本
    (img, path,inp), target = dataset[sample_index]

    print("图像大小:", img.shape)  # Tensor 的 shape
    print("图像路径:", path)
    print("标签:", target)
    print("path" , inp)

    # 可以尝试 DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for (imgs, paths, inp), targets in loader:
        print("Batch 图像大小:", imgs.shape)  # [batch, 3, H, W]
        print("Batch 路径:", paths)
        print("Batch 标签:", targets)
        print("inp", inp)
        break  # 只测试第一个 batch