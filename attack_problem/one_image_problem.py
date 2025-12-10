
import torch
import numpy as np

class SingleImageProblem:
    """
    Represents an adversarial attack problem for a single image.

    Optimization Goal
    -----------------
    The optimizer minimizes the fitness value.
    A fitness value of 0 indicates an optimal solution, a successful attack.

    Attributes
    ----------
    image : np.ndarray
        The original image to attack.
    target_label : int
        The target label we want to atain
    ml_model : torch.nn.Module
        The machine learning model used for evaluation.
    max_evaluation : int
        Maximum allowed number of fitness evaluations.
    """
    def __init__(self, config):
        
        self.__ml_model = config["ml_model"]
        self.__image = config["image"]
        self.__y_target = config["y_target"]
        self.__image_size = config.get("image_size", 448)
        self.__epsilon = config.get("epsilon", 77.62)
        self.__max_evaluation = config.get("max_eval", 10000)
        self.__target_label = np.argwhere(self.__y_target > 0)
        self.__evaluations= 0

        nchannels=3
        self.__dimension = (self.__image_size ** 2) * nchannels
        self.__x_range = [(-1, 1)] * self.__dimension


    @property
    def evaluations(self):
        """只读属性，返回 evaluations"""
        return self.__evaluations

    @property
    def epsilon(self):
        """只读属性，返回 epsilon"""
        return self.__epsilon

    # 只读属性：max_evaluation
    @property
    def max_evaluation(self):
        """只读属性，返回 max_evaluation"""
        return self.__max_evaluation

    def get_dimension(self):
        """返回 dimension = (image_size^2) * 3"""
        return self.__dimension

    def get_x_range(self):
        """返回 dimension 维，每维范围为 (-1, 1)"""
        return self.__x_range

    def evaluate(self, x, effective=True):
        """
        Evaluate the fitness of the given perturbations x on the current image.

        Parameters
        ----------
        x : numpy.ndarray
            The perturbations to evaluate, shape = (batch_size, C, H, W).

        Returns
        -------
        fitness : numpy.ndarray
            整体攻击目标函数值（所有 label 的分量求和后的总 fitness 值）。
            Shape: (batch_size, 1)
            表示每个样本的最终 fitness，用于黑盒优化器的目标值。

        fit : numpy.ndarray
            每个类别的攻击损失分量（逐 label 的分量贡献）。
            Shape: (batch_size, num_classes)
            表示每个样本、每个类别对 fitness 的贡献。
            
        """
        
        if effective:
            if self.__evaluations + len(x) > self.__max_evaluation:
                #raise Exception("Exceeded maximum number of evaluations.")
                return None, None
            self.__evaluations += len(x)
            
        with torch.no_grad():
            if torch.cuda.is_available():
                predict = self.__ml_model(torch.tensor(np.clip(np.tile(self.__image, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.__image.shape) , 0.,
                                                          1.), dtype=torch.float32).cuda()).cpu()
            else:
                predict = self.__ml_model(torch.tensor(np.clip(np.tile(self.__image, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.__image.shape) , 0.,
                                                          1.), dtype=torch.float32))
        p = np.copy(predict)
        q = np.zeros(p.shape)+0.5
        fit = p-q
        fit[:,self.__target_label]=-fit[:,self.__target_label]
        fit[np.where(fit<0)]=0
        fitness = np.sum(fit,axis=1)
        fitness = fitness[:, np.newaxis]

        return fitness, fit
    
    def getFeasible(self, x):
        """
        Check the feasibility of the perturbations x based on L2 norm constraint.

        Parameters
        ----------
        x : numpy.ndarray
            The perturbations to check, shape = (batch_size, C, H, W).

        Returns
        -------
        feasible : numpy.ndarray
            A boolean array indicating whether each perturbation is feasible.
            Shape: (batch_size,)
        """
        batch_norm = self.l2_norm(x)
        feasible = np.array(batch_norm) <= self.__epsilon
        return feasible

    def l2_norm(self, x):

        # 对 batch 中每个扰动计算 L2
        batch_norm = [vector_norm(r) for r in x]
        return batch_norm
    
    
def vector_norm(r):
    """
    返回向量 r 的 L2 范数：||r||_2
    自动 flatten，再求 norm
    """
    return np.linalg.norm(r.flatten())
    

        