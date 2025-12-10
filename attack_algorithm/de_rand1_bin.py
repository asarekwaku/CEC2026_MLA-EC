import numpy as np
import random
from attack_algorithm.attack_algorithm_base import AttackAlgorithmBase
from attack_problem.one_image_problem import SingleImageProblem

class DE_RAND1(AttackAlgorithmBase):

    def __init__(self, config):
        super().__init__(config)
        # 默认参数
        self.pop_size = config.get("pop_size", 20)
        self.F = config.get("F", 0.5)
        self.CR = config.get("CR", 0.9)
        self.eps = config.get("eps", 0.01)

    # =====================================================
    #   evolve: MLDE 的 DE 主循环
    # =====================================================
    def evolve(self, problem: SingleImageProblem):
        rnd = self.rnd
        return DE(self.pop_size, self.F, self.CR, self.eps, problem, rnd)


def mutation(pop, F, rnd):
    """经典 DE/rand/1 变异"""
    pop_size = len(pop)
    dim = pop.shape[1]
    mutant = np.zeros_like(pop)
    for i in range(pop_size):
        idxs = list(range(pop_size))
        idxs.remove(i)
        a, b, c = rnd.sample(idxs, 3)
        mutant[i] = pop[a] + F * (pop[b] - pop[c])
    return mutant


def crossover(pop, mutant, CR, rnd):
    """二进制交叉"""
    pop_size, dim = pop.shape
    trial = np.copy(pop)
    for i in range(pop_size):
        jrand = rnd.randint(0, dim - 1)
        for j in range(dim):
            if rnd.random() < CR or j == jrand:
                trial[i, j] = mutant[i, j]
    return trial


def select(pop, trial, fitness, problem: SingleImageProblem):
    trial_fitness, trial_fit = problem.evaluate(trial)
    new_pop = np.copy(pop)
    new_fitness = np.copy(fitness)
    for i in range(len(pop)):
        if trial_fitness[i] < fitness[i]:
            new_pop[i] = trial[i]
            new_fitness[i] = trial_fitness[i]
    return new_pop, new_fitness


def DE(pop_size, F, CR, eps, problem: SingleImageProblem, rnd: random):
    """经典 DE 主循环"""
    generation_save = []

    dim = problem.get_dimension()
    x_range = problem.get_x_range()

    # 初始化种群
    pop = np.zeros((pop_size, dim))
    for i in range(dim):
        low, high = x_range[i]
        pop[:, i] = np.array([rnd.uniform(low, high) * eps for _ in range(pop_size)])

    fitness, fit = problem.evaluate(pop)
    generation_save.append(np.min(fitness))

    while problem.evaluations < problem.max_evaluation:
        mutant = mutation(pop, F, rnd)
        trial = crossover(pop, mutant, CR, rnd)
        pop, fitness = select(pop, trial, fitness, problem)
        best_idx = np.argmin(fitness)
        pop_r= problem.l2_norm(pop) 
        r_min= np.min(pop_r)    
        print(f"Evaluation:{problem.evaluations}, Best fitness:{fitness[best_idx]}, Best radius:{r_min}")
        print(f"Evaluation: {problem.evaluations}, Best fitness: {fitness[best_idx]}")
        generation_save.append(fitness[best_idx])
        # 找到完全成功的解
        if np.any(fitness == 0):
            return pop[np.argwhere(fitness == 0)[0][0]]

    return pop[np.argmin(fitness)]
