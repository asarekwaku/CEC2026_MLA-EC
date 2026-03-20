import random
import numpy as np
import time
from attack_problem.problem import EvolutionaryAttackProblem
from attack_algorithm.corse_sade import CoRSE_SaDE

def run_validation_test():
    proConfig = {
        "ml_model_name": "mlliw",
        "dataset_name": "voc2007",
        "target_type": "random",
        "epsilon": 77.596,
        "max_eval": 10000,
        "data_dir": "/Users/kwakuasare/Downloads/MLAE_cec_data"
    }
    
    seeds = [1, 2, 3]
    print(f"Starting Validation Test using VOC2007, ML-LIW over seeds: {seeds}")
    
    for seed in seeds:
        print(f"\n--- Testing Seed {seed} ---")
        algConfig = {
            "pop_size": 40,
            "rnd": random.Random(seed)
        }
        
        problem = EvolutionaryAttackProblem(proConfig)
        algo = CoRSE_SaDE(algConfig)
        
        start_time = time.time()
        problem.attack(algo)
        end_time = time.time()
        
        # NOTE: problem._test_ids should be limited to 20 for quick validation.
        # This requires manually adjusting EvolutionaryAttackProblem initialization 
        # or limiting the testing loop in problem.attack()
        
        print(f"Seed {seed} complete in {end_time - start_time:.2f}s")
        print(f"Attack Rate: {problem.attack_rate()}")

if __name__ == '__main__':
    run_validation_test()
