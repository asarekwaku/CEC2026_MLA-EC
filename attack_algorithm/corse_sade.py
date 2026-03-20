import numpy as np
import random
import torch
import torch.nn.functional as F
import copy
from attack_algorithm.attack_algorithm_base import AttackAlgorithmBase
from attack_problem.one_image_problem import SingleImageProblem

class Individual:
    def __init__(self, z, F_i, CR_i, r=None):
        self.z = z
        self.F = F_i
        self.CR = CR_i
        self.r = r
        self.fitness = None
        self.fit = None
        self.score = float('inf')
        self.Lp = float('inf')
        self.Ls = float('inf')

class SubspaceSpec:
    def __init__(self, stage_type, **kwargs):
        self.stage_type = stage_type
        if self.stage_type == "latent_grid":
            self.grid_hw = kwargs.get("grid_hw", 56)
            self.channels = 3
        elif self.stage_type == "dct_lowfreq":
            self.freq_hw = kwargs.get("freq_hw", 32)
            self.channels = 3
        elif self.stage_type == "patch_set":
            self.num_patches = kwargs.get("num_patches", 8)
            
    def latent_dimension(self):
        if self.stage_type == "latent_grid":
            return self.channels * self.grid_hw * self.grid_hw
        elif self.stage_type == "dct_lowfreq":
            return self.channels * self.freq_hw * self.freq_hw
        elif self.stage_type == "patch_set":
            return self.num_patches * 6 # x, y, size, aR, aG, aB
        return 0
        
    def decode(self, z):
        if self.stage_type == "latent_grid":
            z_img = z.reshape(self.channels, self.grid_hw, self.grid_hw)
            t = torch.tensor(z_img).unsqueeze(0).float()
            upsampled = F.interpolate(t, size=(448, 448), mode='bilinear', align_corners=False)
            return upsampled.squeeze(0).numpy().flatten()
            
        elif self.stage_type == "dct_lowfreq":
            try:
                from scipy.fft import idctn
                coeffs = np.zeros((3, 448, 448))
                z_img = z.reshape(3, self.freq_hw, self.freq_hw)
                coeffs[:, :self.freq_hw, :self.freq_hw] = z_img
                delta_full = idctn(coeffs, axes=(1, 2), norm='ortho')
                return delta_full.flatten()
            except ImportError:
                # Fallback to latent grid if missing scipy
                z_img = z.reshape(3, self.freq_hw, self.freq_hw)
                t = torch.tensor(z_img).unsqueeze(0).float()
                upsampled = F.interpolate(t, size=(448, 448), mode='bilinear', align_corners=False)
                return upsampled.squeeze(0).numpy().flatten()
                
        elif self.stage_type == "patch_set":
            delta = np.zeros((3, 448, 448))
            z_reshaped = z.reshape(self.num_patches, 6)
            for p in z_reshaped:
                x = int((p[0] + 1) / 2 * 448)
                y = int((p[1] + 1) / 2 * 448)
                s = int((p[2] + 1) / 2 * 64) + 4
                aR, aG, aB = p[3], p[4], p[5]
                x = np.clip(x, 0, 447)
                y = np.clip(y, 0, 447)
                s_x = min(s, 448 - x)
                s_y = min(s, 448 - y)
                if s_x > 0 and s_y > 0:
                    delta[0, y:y+s_y, x:x+s_x] += aR
                    delta[1, y:y+s_y, x:x+s_x] += aG
                    delta[2, y:y+s_y, x:x+s_x] += aB
            return delta.flatten()
        return np.zeros(448*448*3)

def IdentifyPrimaryLabels(problem: SingleImageProblem):
    dim = problem.get_dimension()
    x0 = np.zeros((1, dim))
    fitness0, fit0 = problem.evaluate(x0, effective=True)
    if fitness0 is None or len(fitness0) == 0:
        return set(range(dim)), None, None
    primary = {k for k in range(len(fit0[0])) if fit0[0][k] > 0}
    if len(primary) == 0:
        primary = set()
    return primary, fitness0[0], fit0[0]

def ComputeLossDecomposition(fit_vec, primary):
    L_primary = sum(fit_vec[k] for k in primary)
    L_secondary = sum(fit_vec) - L_primary
    return L_primary, L_secondary

def LambdaSchedule(eval_used, max_eval, lambda_min, lambda_max):
    t = eval_used / max_eval
    return lambda_min + (lambda_max - lambda_min) * (t**2)

def ClipAndProject(r_flat, x_range, epsilon):
    lows = np.array([x[0] for x in x_range])
    highs = np.array([x[1] for x in x_range])
    r_flat = np.clip(r_flat, lows, highs)
    norm = np.linalg.norm(r_flat)
    if norm > epsilon:
        r_flat = r_flat * (epsilon / norm)
    return r_flat

def SafeEvaluateBatch(problem: SingleImageProblem, R_batch_flat):
    remaining = problem.max_evaluation - problem.evaluations
    if remaining <= 0:
        return None, None, 0
    B = min(len(R_batch_flat), remaining)
    fitness, fit = problem.evaluate(np.array(R_batch_flat[0:B]), effective=True)
    return fitness, fit, B

def ComputeSelectionScore(fitness_scalar, fit_vec, primary, lambda_t):
    Lp, Ls = ComputeLossDecomposition(fit_vec, primary)
    score = Lp + lambda_t * Ls
    return score, Lp, Ls

def SampleJDEParams(F_i, CR_i, rnd, tau1, tau2, F_l, F_u):
    F_new = F_l + rnd.random() * (F_u - F_l) if rnd.random() < tau1 else F_i
    CR_new = rnd.random() if rnd.random() < tau2 else CR_i
    return F_new, CR_new

def UpdateSuccessHistory(history, successful_F, successful_CR, alpha):
    if successful_F:
        history['F_mean'] = (1 - alpha) * history['F_mean'] + alpha * np.mean(successful_F)
    if successful_CR:
        history['CR_mean'] = (1 - alpha) * history['CR_mean'] + alpha * np.mean(successful_CR)
    return history

def AntitheticProbe(problem, best_z, spec, sigma, primary, lambda_t, x_range, epsilon):
    dim_latent = spec.latent_dimension()
    u = np.random.randn(dim_latent)
    u = u / (np.linalg.norm(u) + 1e-8)
    
    z_plus = best_z + sigma * u
    z_minus = best_z - sigma * u
    
    r_plus = ClipAndProject(spec.decode(z_plus), x_range, epsilon)
    r_minus = ClipAndProject(spec.decode(z_minus), x_range, epsilon)
    
    fitness, fit, used = SafeEvaluateBatch(problem, [r_plus, r_minus])
    if used < 2: return None
    
    score_plus, _, _ = ComputeSelectionScore(fitness[0], fit[0], primary, lambda_t)
    score_minus, _, _ = ComputeSelectionScore(fitness[1], fit[1], primary, lambda_t)
    
    if score_plus < score_minus:
        return {'direction': u, 'winner': z_plus, 'score': score_plus, 'r': r_plus, 'fit': fit[0], 'fitness': fitness[0]}
    else:
        return {'direction': -u, 'winner': z_minus, 'score': score_minus, 'r': r_minus, 'fit': fit[1], 'fitness': fitness[1]}

def Polisher(problem, r_best, primary, mode, steps, lambda_min, lambda_max, best_score_polish):
    dim = problem.get_dimension()
    x_range = problem.get_x_range()
    eps = problem.epsilon
    fit_best_cached = None
    
    for t in range(1, steps + 1):
        if problem.evaluations >= problem.max_evaluation:
            break
            
        lambda_t = LambdaSchedule(problem.evaluations, problem.max_evaluation, lambda_min, lambda_max)
        
        if mode == "square":
            patch_size = max(4, int(64 * (1 - t/steps)))
            amp = max(0.01, 0.2 * (1 - t/steps))
            
            delta = np.zeros((3, 448, 448))
            c = random.randint(0, 2)
            x = random.randint(0, 448 - patch_size)
            y = random.randint(0, 448 - patch_size)
            sign = 1 if random.random() > 0.5 else -1
            delta[c, y:y+patch_size, x:x+patch_size] = sign * amp
            
            proposal = r_best + delta.flatten()
            proposal = ClipAndProject(proposal, x_range, eps)
            
            fitness, fit, used = SafeEvaluateBatch(problem, [proposal])
            if used == 0: break
            
            score_prop, _, _ = ComputeSelectionScore(fitness[0], fit[0], primary, lambda_t)
            if score_prop <= best_score_polish:
                r_best = proposal
                best_score_polish = score_prop
                fit_best_cached = fit[0]
                if fitness[0] == 0:
                    return r_best, fit_best_cached
    return r_best, fit_best_cached

class CoRSE_SaDE(AttackAlgorithmBase):
    def __init__(self, config):
        super().__init__(config)
        self.pop_size = config.get("pop_size", 40)
        self.tau1 = config.get("tau1", 0.1)
        self.tau2 = config.get("tau2", 0.1)
        self.F_l = config.get("F_l", 0.1)
        self.F_u = config.get("F_u", 0.9)
        self.lambda_min = config.get("lambda_min", 0.05)
        self.lambda_max = config.get("lambda_max", 1.0)
        
        self.stages = config.get("stages", [
            {"type": "latent_grid", "grid_hw": 56, "budget_frac": 0.55, "alpha_hist": 0.2},
            {"type": "dct_lowfreq", "freq_hw": 32, "budget_frac": 0.30, "alpha_hist": 0.2},
            {"type": "patch_set", "num_patches": 8, "budget_frac": 0.10, "alpha_hist": 0.2}
        ])
        
        self.beta_m = config.get("beta_m", 0.9)
        self.probe_every = config.get("probe_every", 5)
        self.probe_sigma = config.get("probe_sigma", 0.1)
        
        self.stall_gens = config.get("stall_gens", 10)
        self.restart_frac = config.get("restart_frac", 0.5)
        self.elite_frac = config.get("elite_frac", 0.1)
        
        self.polish_mode = config.get("polish_mode", "square")
        self.polish_steps = config.get("polish_steps", 500)

    def evolve(self, problem: SingleImageProblem):
        rnd = self.rnd
        dim = problem.get_dimension()
        x_range = problem.get_x_range()
        epsilon = problem.epsilon
        max_eval = problem.max_evaluation
        
        primary, fitness0, fit0 = IdentifyPrimaryLabels(problem)
        if fitness0 == 0:
            return np.zeros(dim)
            
        best = None
        best_score = float('inf')
        m = None
        
        history = {'F_mean': 0.5, 'CR_mean': 0.9}
        
        for stage in self.stages:
            stage_budget = int(stage['budget_frac'] * max_eval)
            stage_start_eval = problem.evaluations
            
            spec = SubspaceSpec(stage["type"], **stage)
            dim_latent = spec.latent_dimension()
            m = np.zeros(dim_latent)
            
            pop = []
            for i in range(self.pop_size):
                z = np.random.uniform(-1, 1, dim_latent)
                ind = Individual(z, history['F_mean'], history['CR_mean'])
                pop.append(ind)
                
            R_batch = []
            for ind in pop:
                r = spec.decode(ind.z)
                r = ClipAndProject(r, x_range, epsilon)
                ind.r = r
                R_batch.append(r)
                
            fitness, fit, used = SafeEvaluateBatch(problem, R_batch)
            if used == 0: break
            
            for i in range(used):
                lambda_t = LambdaSchedule(problem.evaluations, max_eval, self.lambda_min, self.lambda_max)
                pop[i].fitness = fitness[i]
                pop[i].fit = fit[i]
                pop[i].score, pop[i].Lp, pop[i].Ls = ComputeSelectionScore(fitness[i], fit[i], primary, lambda_t)
                
                if pop[i].score < best_score:
                    best = copy.deepcopy(pop[i])
                    best_score = pop[i].score
            
            gen = 0
            last_improve_gen = 0
            break_all = False
            
            while problem.evaluations - stage_start_eval < stage_budget:
                if problem.evaluations >= max_eval:
                    break_all = True
                    break
                    
                gen += 1
                
                if gen % self.probe_every == 0:
                    valid_pop = [x for x in pop if x.score != float('inf')]
                    if valid_pop:
                        stage_best = min(valid_pop, key=lambda x: x.score)
                        lambda_t = LambdaSchedule(problem.evaluations, max_eval, self.lambda_min, self.lambda_max)
                        probe = AntitheticProbe(problem, stage_best.z, spec, self.probe_sigma, primary, lambda_t, x_range, epsilon)
                    if probe is not None:
                        m = self.beta_m * m + (1 - self.beta_m) * probe['direction']
                        if probe['score'] < best_score:
                            best.z = probe['winner']
                            best.r = probe['r']
                            best.score = probe['score']
                            best.fitness = probe['fitness']
                            best.fit = probe['fit']
                            best_score = probe['score']
                            last_improve_gen = gen
                            if probe['fitness'] == 0:
                                break_all = True
                                break
                            
                trial_pop = []
                successful_F = []
                successful_CR = []
                
                for i in range(min(self.pop_size, len(pop))):
                    F_i, CR_i = SampleJDEParams(pop[i].F, pop[i].CR, rnd, self.tau1, self.tau2, self.F_l, self.F_u)
                    
                    idxs = list(range(len(pop)))
                    idxs.remove(i)
                    a, b, c = random.sample(idxs, 3)
                    
                    v = pop[a].z + F_i * (pop[b].z - pop[c].z)
                    v = v + 0.05 * m 
                    
                    u = np.copy(pop[i].z)
                    jrand = random.randint(0, dim_latent - 1)
                    for j in range(dim_latent):
                        if random.random() < CR_i or j == jrand:
                            u[j] = v[j]
                            
                    r_u = ClipAndProject(spec.decode(u), x_range, epsilon)
                    trial_ind = Individual(u, F_i, CR_i, r_u)
                    trial_pop.append(trial_ind)
                    
                R_batch = [ind.r for ind in trial_pop]
                fitness, fit, used = SafeEvaluateBatch(problem, R_batch)
                if used == 0:
                    break_all = True
                    break
                    
                lambda_t = LambdaSchedule(problem.evaluations, max_eval, self.lambda_min, self.lambda_max)
                
                for i in range(used):
                    trial_pop[i].fitness = fitness[i]
                    trial_pop[i].fit = fit[i]
                    trial_pop[i].score, trial_pop[i].Lp, trial_pop[i].Ls = ComputeSelectionScore(fitness[i], fit[i], primary, lambda_t)
                    
                    if trial_pop[i].score <= pop[i].score:
                        pop[i] = trial_pop[i]
                        successful_F.append(pop[i].F)
                        successful_CR.append(pop[i].CR)
                        
                        if pop[i].score < best_score:
                            best = copy.deepcopy(pop[i])
                            best_score = pop[i].score
                            last_improve_gen = gen
                            
                    if trial_pop[i].fitness == 0:
                        best = copy.deepcopy(trial_pop[i])
                        best_score = trial_pop[i].score
                        break_all = True
                        break
                        
                if break_all: break
                
                history = UpdateSuccessHistory(history, successful_F, successful_CR, stage.get('alpha_hist', 0.2))
                
                if gen - last_improve_gen >= self.stall_gens:
                    pop.sort(key=lambda x: x.score)
                    keep_idx = int(self.pop_size * self.elite_frac)
                    replace_idx = int(self.pop_size * self.restart_frac)
                    for re_i in range(keep_idx, keep_idx + replace_idx):
                        if re_i < len(pop):
                            z = np.random.uniform(-1, 1, dim_latent)
                            pop[re_i].z = z
                    last_improve_gen = gen
                    
            if break_all: break
            
        if best is None:
            return np.zeros(dim)
            
        best_r = best.r
        if best.fitness == 0:
            return best_r
            
        best_r, _ = Polisher(problem, best_r, primary, self.polish_mode, self.polish_steps, self.lambda_min, self.lambda_max, best_score)
        return best_r
