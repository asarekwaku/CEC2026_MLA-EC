import torch
import os
import numpy as np
import ml_model.dataset_generate as dataset_generate
from ml_model import model_util
from attack_algorithm.attack_algorithm_base import AttackAlgorithmBase
from attack_problem.one_image_problem import *
from attack_problem.util import get_phase_name, get_adv_dir




class EvolutionaryAttackProblem:
      
    def __init__(self, proConfig):

        # ---- 读取配置 ----
        self.__model_name = proConfig["ml_model_name"]
        self.__dataset_name = proConfig["dataset_name"]

        self.__target_type = proConfig.get("target_type", 'random')
        self.__image_size = proConfig.get("image_size", 448)
        self.__epsilon = proConfig.get("epsilon", 77.596)
        self.__max_evaluation = proConfig.get("max_eval", 10000)
        self.__batch_size = proConfig.get("batch_size", 10)
        self.__workers = proConfig.get("workers", 4)
        
        self.__datadir = proConfig["data_dir"]
        

        # ---- 生成数据集 ----
        phase_name = get_phase_name(
            self.__dataset_name,
            self.__model_name,
            self.__target_type
        )
        self.dataset = dataset_generate.generateDataSet(
            self.__datadir,
            self.__dataset_name,
            phase_name,
            self.__image_size
        )

        # ---- 模型 ----
        self.__model = model_util.generateModel(self.__datadir, self.__model_name, self.dataset)
        self.use_gpu = torch.cuda.is_available()

        self.__model.eval()
        if self.use_gpu:
            self.__model = self.__model.cuda()

        # ---- DataLoader ----
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.__batch_size,
            shuffle=False,
            num_workers=self.__workers
        )

        # ---- 载入 y.npy / y_target.npy ----
        pathdir = get_adv_dir(self.dataset, phase_name)
        self.y_target_path = os.path.join(pathdir, self.__target_type + '_y_target.npy')
        self.y_npy_path = os.path.join(pathdir, self.__target_type + '_y.npy')

        self.y_target = np.load(self.y_target_path)     # 需 load
        self.y = np.load(self.y_npy_path)
        
        self.success_count = 0
        self.total_count = 0
        self.save_image_r=[]
        
        
        
    def attack_rate(self):  
        if self.total_count ==0:
            return 0.0
        return self.success_count / self.total_count


    def attack(self, alg: AttackAlgorithmBase):

        batch_size = self.__batch_size

        # 注意这里应该使用 self.loader
        for i, (input, target) in enumerate(self.loader):

            print(f'{i} generator data, length is {len(input[0])}')

            # input[0] shape: (B, C, H, W)
            x_list = input[0].cpu().numpy()

            # 根据 batch index 提取目标标签
            begin = i * batch_size
            end = begin + len(target)
            y_target_batch = self.y_target[begin:end]

            # 每张图单独攻击
            for j in range(len(x_list)):

                config = {
                    "ml_model": self.__model,
                    "image": x_list[j],            # numpy (C,H,W)
                    "y_target": y_target_batch[j],
                    "epsilon": self.__epsilon,
                    "max_eval": self.__max_evaluation
                }

                # 创建单图攻击问题
                problem = SingleImageProblem(config)

                # 调用算法 evolve()
                r = alg.evolve(problem)
                best = r[None, :]
                fitness, _ = problem.evaluate(best,False)
                
                if fitness==0 and vector_norm(r)<= self.__epsilon:
                    self.success_count += 1
                self.total_count += 1
                self.save_image_r.append((config.get('image'), r[0]))
                norm_r= vector_norm(r)
                print(f"success_count={self.success_count}, total_count={self.total_count}")
                print(f"fitness={fitness}, r={norm_r}")

