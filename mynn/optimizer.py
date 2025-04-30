# 优化器模块，包含基础优化器类和具体优化算法实现
from abc import abstractmethod
import numpy as np


class Optimizer:
    """优化器基类，定义优化器的通用接口"""
    
    def __init__(self, init_lr, model) -> None:
        """
        初始化优化器
        
        参数:
            init_lr: float - 初始学习率
            model: Model - 需要优化的模型对象
        """
        self.init_lr = init_lr  # 基础学习率
        self.model = model      # 待优化的模型

    @abstractmethod
    def step(self):
        """抽象方法，执行一步参数更新"""
        pass


class SGD(Optimizer):
    """随机梯度下降优化器(Stochastic Gradient Descent)"""
    
    def __init__(self, init_lr, model):
        """
        初始化SGD优化器
        
        参数:
            init_lr: float - 初始学习率
            model: Model - 需要优化的模型对象
        """
        super().__init__(init_lr, model)
    
    def step(self):
        """执行一步SGD参数更新"""
        # 遍历模型中的所有层
        for layer in self.model.layers:
            # 只优化可训练层
            if layer.optimizable == True:
                # 遍历层中的所有参数
                for param_name in layer.params.keys():
                    # 执行梯度下降更新
                    layer.params[param_name] = layer.params[param_name] - self.init_lr * layer.grads[param_name]


class MomentGD(Optimizer):
    """带动量的梯度下降优化器(Momentum Gradient Descent)"""
    
    def __init__(self, init_lr, model, mu):
        """
        初始化带动量的梯度下降优化器
        
        参数:
            init_lr: float - 初始学习率
            model: Model - 需要优化的模型对象
            mu: float - 动量系数(通常取值0.9左右)
        """
        super().__init__(init_lr, model)
        self.mu = mu  # 动量系数
        self.velocity = {}  # 用于存储各参数的动量(速度)
        
        # 初始化各层的动量缓冲区
        for layer_idx, layer in enumerate(model.layers):
            if layer.optimizable:
                self.velocity[layer_idx] = {}
                # 为每个参数初始化动量(速度)为0
                for param_name in layer.params.keys():
                    self.velocity[layer_idx][param_name] = np.zeros_like(layer.params[param_name])
    
    def step(self):
        """执行一步带动量的参数更新"""
        # 遍历模型中的所有层
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                # 遍历层中的所有参数
                for param_name in layer.params.keys():
                    # 更新动量(速度)
                    self.velocity[layer_idx][param_name] = self.mu * self.velocity[layer_idx][param_name] - self.init_lr * layer.grads[param_name]
                    
                    
                    # 使用动量更新参数
                    layer.params[param_name] += self.velocity[layer_idx][param_name]
