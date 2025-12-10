from PIL import Image
import torchvision.transforms as transforms
import torch
import os
import numpy as np
from tqdm import tqdm
from dataset.basic_dataset import BaseClassificationDataset 
from pathlib import Path
import shutil
import numpy as np
from PIL import Image





def evaluate_model(model, loader, use_gpu):
    tqdm.monitor_interval = 0

    output = []
    y = []
    test_loader = tqdm(test_loader, desc='Test')
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            x = input[0]
            if use_gpu:
                x = x.cuda()
            o = model(x).cpu().numpy()
            output.extend(o)
            y.extend(target.cpu().numpy())
        output = np.asarray(output)
        y = np.asarray(y)

    pred = (output >= 0.5) + 0
    y[y == -1] = 0

    from utils import evaluate_metrics
    metric = evaluate_metrics.evaluate(y, output, pred)
    print(metric)



def judgeAttackable(y_i, target_type):
    """
    检查标签矩阵是否包含可行的攻击目标。

    :param y: numpy.ndarray, 标签矩阵，元素为 {0, 1}
    :param target_type: str, 标签处理类型，可选 'hide_single', 'hide_all', 'random'
        - 'hide_single' : 每个样本随机反转一个正标签
        - 'hide_all'    : 所有正标签变成 -1
        - 'random'      : 每个样本随机反转一个正标签和一个负标签
    :return: numpy.ndarray, 布尔数组，表示每个样本是否包含可行的攻击目标
    """ 
    
    y_i = y_i.copy()
    
    # 将 0 转为 -1
    y_i[y_i == 0] = -1
    
    feasibel_flag = False
    if target_type == 'hide_single':
        # 每个样本随机反转一个正标签
        pos_idx = np.where(y_i == 1)[0]
        if len(pos_idx) > 0:
            feasibel_flag= True


    elif target_type == 'hide_all':
        pos_idx = np.where(y_i == 1)[0]
        if len(pos_idx) > 0:
            feasibel_flag= True

    elif target_type == 'random':
        # 每个样本随机反转一个正标签和一个负标签
        pos_idx = np.where(y_i == 1)[0]
        neg_idx = np.where(y_i == -1)[0]
        
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            feasibel_flag= True

    return feasibel_flag






def get_target_label(y, target_type, rnd):
    """
    修改标签矩阵，根据 target_type 隐藏或随机反转标签。

    :param y: numpy.ndarray, 标签矩阵，元素为 {0, 1}
    :param target_type: str, 标签处理类型，可选 'hide_single', 'hide_all', 'random'
        - 'hide_single' : 每个样本随机反转一个正标签
        - 'hide_all'    : 所有正标签变成 -1
        - 'random'      : 每个样本随机反转一个正标签和一个负标签
    :param rnd: random 模块实例，用于随机选择
    :return: numpy.ndarray, 处理后的标签矩阵，元素为 {-1, 1}
    """
    y = y.copy()
    
    # 将 0 转为 -1
    y[y == 0] = -1

    if target_type == 'hide_single':
        # 每个样本随机反转一个正标签
        for i, y_i in enumerate(y):
            pos_idx = np.where(y_i == 1)[0]
            if len(pos_idx) > 0:
                idx = rnd.choice(pos_idx)
                y[i, idx] = -y[i, idx]

    elif target_type == 'hide_all':
        # 将所有正标签变成 -1
        y[y == 1] = -1

    elif target_type == 'random':
        # 每个样本随机反转一个正标签和一个负标签
        for i, y_i in enumerate(y):
            pos_idx = np.where(y_i == 1)[0]
            neg_idx = np.where(y_i == -1)[0]

            if len(pos_idx) > 0:
                idx = rnd.choice(pos_idx)
                y[i, idx] = -y[i, idx]

            if len(neg_idx) > 0:
                idx = rnd.choice(neg_idx)
                y[i, idx] = -y[i, idx]

    return y


def get_phase_name(dataset_name,ml_model_name,target_type):
    return ml_model_name +'_'+ dataset_name+"_"+target_type   

def get_adv_dir(dataset:BaseClassificationDataset, phase_name):
    saveDir   = dataset.imageDir( phase_name)    
    # 拼接 CSV 文件路径
    pathdir = os.path.join(saveDir, 'adv_label')
    return pathdir


def gen_adv_file(model, ml_model_name,
                 dataset:BaseClassificationDataset, test_loader,
                 target_type, use_gpu, rnd, max_samples=100):
    print("generiting……")
    tqdm.monitor_interval = 0
    output = []
    image_name_list = []
    y = []
    test_loader = tqdm(test_loader, desc='Test')
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            x = input[0]
            if use_gpu:
                x = x.cuda()
            o = model(x).cpu().numpy()
            output.extend(o)
            y.extend(target.cpu().numpy())
            image_name_list.extend(list(input[1]))
        output = np.asarray(output)
        y = np.asarray(y)
        image_name_list = np.asarray(image_name_list)

    # choose x which can be well classified and contains two or more label to prepare attack
    pred = (output >= 0.5) + 0
    y[y==-1] = 0
    true_idx = []
    
    
    
    count = 0
    for i in range(len(pred)):
        sum_y=np.sum(y[i])
        if (y[i] == pred[i]).all() and judgeAttackable(y[i], target_type) and count < max_samples:
            true_idx.append(i)
            count += 1
    # adv_image_name_list = image_name_list[true_idx]
    adv_y = y[true_idx]
    y = y[true_idx]
    y_target = get_target_label(adv_y, target_type,rnd)
    y_target[y_target==0] = -1
    y[y==0] = -1
    
#    phase_name= ml_model_name +'_'+ dataset.name()+"_"+target_type
    phase_name=get_phase_name(dataset.name(),ml_model_name,target_type)
    dataset.outputFileter(true_idx, phase_name)


    originDir = dataset.originImageDir()
    saveDir   = dataset.imageDir(phase_name)

    os.makedirs(saveDir, exist_ok=True)

    for it in true_idx:
        src = os.path.join(originDir, dataset.imageName(it))
        dst = os.path.join(saveDir, dataset.imageName(it))
        shutil.copy(src, dst)
        
    
    
    # 拼接 CSV 文件路径
#    pathdir = os.path.join(saveDir, 'adv_label')
    
    pathdir=get_adv_dir(dataset,phase_name)
    pathdir = str(pathdir)
    if not os.path.exists(pathdir):  # create dir if necessary
        os.makedirs(pathdir)
    
    # save target y and ground-truth y to prepare attack
    # value is {-1,1}
    np.save(os.path.join(pathdir, target_type+'_y_target.npy'), y_target)
    np.save(os.path.join(pathdir, target_type+'_y.npy'), y)
#    np.save('../adv_save/mlgcn/voc2007/y.npy', y)




def save_adv_image(img, r, path):
    adv = img + r                     # (C,H,W)
    adv = np.clip(adv, 0, 1)          # 防止越界

    adv = (adv * 255).astype(np.uint8)
    adv = adv.transpose(1, 2, 0)      # (H,W,C)

    Image.fromarray(adv).save(path)