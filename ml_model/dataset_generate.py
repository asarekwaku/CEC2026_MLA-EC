import torchvision.transforms as transforms
from dataset.coco import COCO2014
from dataset.nuswide import NusWide
from dataset.voc import Voc2007Classification,Voc2012Classification
from ml_model.util import Warp
from ml_model.util import MultiScaleCrop

import os

def _getDefaultTransform(image_size):
    data_transforms = transforms.Compose([
        Warp(image_size),
        transforms.ToTensor(),
        ])
    return data_transforms


def _getVoc2007Transform(phase_type, image_size):
    return _getDefaultTransform(image_size)


def _getVoc2012Transform(phase_type, image_size):
    return _getDefaultTransform(image_size)


def _getNuswideTransform(phase_type, image_size):
    
    return _getDefaultTransform(image_size)


def _getCocoTransform(phase_type, image_size):
    return _getDefaultTransform(image_size)


def getTransform(dataset, phase_type, image_size):

    # if(phase_type=='val'| phase_type=='test'):
    #     return _getDefaultTransform(image_size)

    if isinstance(dataset, NusWide):
        return _getNuswideTransform(phase_type,image_size)
    elif isinstance(dataset, Voc2007Classification):
        return _getVoc2007Transform(phase_type,image_size)
    elif isinstance(dataset, Voc2012Classification):
        return _getVoc2012Transform(phase_type,image_size)
    elif isinstance(dataset, COCO2014):
        return _getCocoTransform(phase_type,image_size)
    else:
        raise ValueError("Unsupported dataset type")



def getTransformByName(dataset_name, phase_type, image_size):

    # if(phase_type=='val'| phase_type=='test'):
    #     return _getDefaultTransform(image_size)

    if dataset_name=='nuswide':
        return _getNuswideTransform(phase_type,image_size)
    elif dataset_name=='voc2007':
        return _getVoc2007Transform(phase_type,image_size)
    elif dataset_name=='voc2012':
        return _getVoc2012Transform(phase_type,image_size)
    elif dataset_name=='coco':
        return _getCocoTransform(phase_type,image_size)
    else:
        raise ValueError("Unsupported dataset type")







def generateDataSet(dataset_dir, dataset_name, phase, image_size):
    """
    根据数据集名称、阶段类型和图像大小生成对应的数据集对象
    """
    transform = getTransformByName(dataset_name, phase, image_size)

    if dataset_name == 'coco':
        
        dataset = COCO2014(root=dataset_dir, transform=transform, phase=phase)
    elif dataset_name == 'voc2007':
        dataset = Voc2007Classification(dataset_dir, transform=transform, phase=phase)
    elif dataset_name == 'voc2012':
        dataset = Voc2012Classification(dataset_dir, transform=transform, phase=phase)
    elif dataset_name == 'nuswide':
        dataset = NusWide(root=dataset_dir, transform=transform, phase=phase)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    return dataset