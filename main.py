from attack_algorithm.mlde import MLDE
from attack_algorithm.de_rand1_bin import DE_RAND1  
from attack_problem.problem import EvolutionaryAttackProblem    

import random

def run_attack():
    # 配置参数  
    proConfig = {
        "ml_model_name": "mlliw",
        "dataset_name": "voc2007",
        "target_type": "random",     # 目标类型：random / hide_single / hide_all
        "image_size": 448,
        # "epsilon": 77.62,            # 控制扰动范围
        "epsilon": 1500.62,            # 控制扰动范围
        "max_eval": 200000,            # 最大评价次数
        "batch_size": 14,             # 你之前要求改成 1
        "workers": 4
    }
    mlde_config = {
        "F": 0.5,
        "CR": 0.9,
        "pop_size": 100,
        "rnd": random.Random(1234)
    }

    problem = EvolutionaryAttackProblem(proConfig)
    mlde = MLDE(mlde_config)
    problem.attack(mlde)


def runMultipleAttacks():
    
    proConfig = {
        "ml_model_name": "mlliw",
        "dataset_name": "voc2007",
        "target_type": "random",     # 目标类型：random / hide_single / hide_all
        "image_size": 448,
        "epsilon": 77.596,            # 控制扰动范围
        "max_eval": 200000,            # 最大评价次数
        "batch_size": 14,             # 你之前要求改成 1
        "workers": 4
    }
    
    master_seed = 20250101   # 固定的主种子 (可复现)
    master_rng = random.Random(master_seed)

    seeds = [master_rng.getrandbits(64) for _ in range(10)]
    print(seeds)

    for run_id in range(10):
        seed = seeds[run_id]
        rnd = random.Random(seed)

        mlde_config = {
            "F": 0.5,
            "CR": 0.9,
            "pop_size": 100,
            "rnd": rnd
        }

        print(f"Run {run_id}, seed = {seed}")
        problem = EvolutionaryAttackProblem(proConfig)
        mlde = MLDE(mlde_config)
        problem.attack(mlde)

import os

if __name__ == '__main__':

    proConfig = {
        "ml_model_name": "mlgcn",   # Multi-label classification model: mlgcn or mlliw
        "dataset_name": "coco",     # Dataset to attack: coco / voc2007 / voc2012 / nuswide
        "target_type": "random",    # Attack type
        "epsilon": 77.596,          # Perturbation limit (L2 norm bound)
        "max_eval": 10000,          # Maximum number of fitness evaluations per image 
        "data_dir": "/home/dyy/code/MLAE_cec_data"  # Path to the MLAE_cec_data directory
    }

    algConfig = {
        "F": 0.5,
        "CR": 0.9,
        "pop_size": 100,
        "rnd": random.Random(1234)
    }

    problem = EvolutionaryAttackProblem(proConfig)
    mlde = DE_RAND1(algConfig)
    problem.attack(mlde)
    print("attack_rate", problem.attack_rate())


