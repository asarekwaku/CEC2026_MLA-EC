
from utils import evaluate_metrics
import torch
import torch.optim as optim
import numpy as np

import ml_model.ml_liw_model.models
import ml_model.ml_gcn_model.models
import os
from dataset import basic_dataset
import torch.optim as optim
import numpy as np
import shutil
import tqdm
def instance_wise_loss(output, y):
    y_i = torch.eq(y, torch.ones_like(y))
    y_not_i = torch.eq(y, -torch.ones_like(y))

    column = torch.unsqueeze(y_i, 2)
    row = torch.unsqueeze(y_not_i, 1)
    truth_matrix = column * row
    column = torch.unsqueeze(output, 2)
    row = torch.unsqueeze(output, 1)
    sub_matrix = column - row
    exp_matrix = torch.exp(-sub_matrix)
    sparse_matrix = exp_matrix * truth_matrix
    sums = torch.sum(sparse_matrix, (1, 2))
    y_i_sizes = torch.sum(y_i, 1)
    y_i_bar_sizes = torch.sum(y_not_i, 1)
    normalizers = y_i_sizes * y_i_bar_sizes
    normalizers_zero = torch.logical_not(torch.eq(normalizers, torch.zeros_like(normalizers)))
    normalizers = normalizers[normalizers_zero]
    sums = sums[normalizers_zero]
    loss = sums / normalizers
    loss = torch.sum(loss)
    return loss

def label_wise_loss(output, y):
    output = torch.transpose(output, 0, 1)
    y = torch.transpose(y,0, 1)
    return instance_wise_loss(output, y)

def criterion(output, y):
    loss = 0.5 * instance_wise_loss(output, y) + label_wise_loss(output, y)
    return loss



def test(model, test_loader, use_cuda):

    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            if use_cuda:
                data, target = data[0].cuda(), target.cuda()
            else:
                data = data[0].cuda()
            output = model(data)
            test_loss += criterion(output, target).item()
            outputs.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

    outputs = np.asarray(outputs)
    targets = np.asarray(targets)
    targets[targets==-1] = 0
    pred = outputs.copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    metrics = evaluate_metrics.evaluate(targets, outputs, pred)
    print(metrics)
    return test_loss


def test_tqdm(model, test_loader, use_cuda):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []

    # tqdm 加进度条
    loop = tqdm(test_loader, desc="Testing", ncols=100)

    with torch.no_grad():
        for data, target in loop:
            if use_cuda:
                data, target = data[0].cuda(), target.cuda()
            else:
                data = data[0].cuda()
            output = model(data)
            test_loss += criterion(output, target).item()
            outputs.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

            # 可选：在进度条显示当前平均损失
            loop.set_postfix(test_loss=test_loss / (len(outputs)/data.size(0)))

    outputs = np.asarray(outputs)
    targets = np.asarray(targets)
    targets[targets==-1] = 0

    pred = outputs.copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    metrics = evaluate_metrics.evaluate(targets, outputs, pred)
    print(metrics)
    return test_loss


def generateModel(dir_path,model_name, dataset: basic_dataset.BaseClassificationDataset):
    # dir_path = os.path.dirname(os.path.abspath(__file__))
    
    if model_name == 'mlliw':
        model = ml_model.ml_liw_model.models.inceptionv3_attack(
            num_classes=dataset.get_number_classes(),
            save_model_path=os.path.join(dir_path, 'checkpoint',model_name, dataset.name(), 'model_best.pth.tar')
        )
    
    elif model_name == 'mlgcn':
        model = ml_model.ml_gcn_model.models.gcn_resnet101_attack(
            num_classes=dataset.get_number_classes(),
            t=0.4,
            adj_file=dataset.adj_filepath(),
            word_vec_file=dataset.inp_filepath(),
            save_model_path=os.path.join(dir_path, 'checkpoint',model_name, dataset.name(), 'model_best.pth.tar')
        )
    
    else:
        raise ValueError('No such model: {}'.format(model_name))
    
    return model



