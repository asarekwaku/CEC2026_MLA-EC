import ml_model.dataset_generate as dataset_generate

from ml_model import model_util
from ml_model.ml_liw_model import train
import torch
from attack_problem.util import gen_adv_file
from attack_problem.util import get_phase_name
import random




def test():
    image_size=448
    batch_size= 10
    workers=4
    dataset= dataset_generate.generateDataSet('./data','nuswide','val', image_size)
    model = model_util.generateModel('mlliw',dataset)
    use_gpu = torch.cuda.is_available()
    model.eval()
    if use_gpu:
        model = model.cuda()
    loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers)
    train.test_tqdm(model, loader,use_gpu)
    


def gen_adv_datas(dataset_name, ml_model_name, target_type,rnd,max_samples=100):
    image_size=448
    batch_size= 10
    workers=4
    dataset= dataset_generate.generateDataSet('./data', dataset_name,'val', image_size)
    model = model_util.generateModel(ml_model_name,dataset)
    use_gpu = torch.cuda.is_available()
    model.eval()
    if use_gpu:
        model = model.cuda()
    loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers)
    

    
    
    gen_adv_file(model, ml_model_name,dataset,loader,target_type,use_gpu, rnd, max_samples)
    


def generateDataSet():
    
    target_type = 'random'
    rnd = random.Random()      # 新建一个随机数对象
    rnd.seed(12345)            # 可选：设置种子保#证可复现
    gen_adv_datas('voc2007','mlliw',target_type, rnd)
    rnd.seed(12345)            # 可选：设置种子保#证可复现
#    test()
    gen_adv_datas('voc2012','mlliw',target_type, rnd)
    rnd.seed(12345)            # 可选：设置种子保#证可复现
    gen_adv_datas('nuswide','mlliw',target_type, rnd)
    rnd.seed(12345)            # 可选：设置种子保#证可复现
    gen_adv_datas('coco','mlliw',target_type, rnd)
    
    rnd.seed(12345)            # 可选：设置种子保#证可复现
    gen_adv_datas('voc2007','mlgcn',target_type, rnd)
    
    rnd.seed(12345)            # 可选：设置种子保#证可复现
    gen_adv_datas('voc2012','mlgcn',target_type, rnd)
    
    rnd.seed(12345)            # 可选：设置种子保#证可复现
    gen_adv_datas('nuswide','mlgcn',target_type, rnd)
    
    rnd.seed(12345)            # 可选：设置种子保#证可复现
    gen_adv_datas('coco','mlgcn',target_type, rnd)
    
    
#    train.test(model, loader,use_gpu)
    


def testModel():
    model_name= 'mlgcn'
    dataset_name= 'nuswide'
    target_type= 'random'
    
    phase_name= get_phase_name(dataset_name,model_name,target_type)
    
    image_size=448
    batch_size= 10
    workers=4
    dataset= dataset_generate.generateDataSet('./data',dataset_name,phase_name, image_size)
    model = model_util.generateModel(model_name,dataset)
    use_gpu = torch.cuda.is_available()
    model.eval()
    if use_gpu:
        model = model.cuda()
    loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers)
    train.test_tqdm(model, loader,use_gpu)
    


if __name__ == '__main__':
    testModel()
