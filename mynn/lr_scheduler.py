from abc import abstractmethod
import numpy as np


class scheduler():
    """学习率调度器基类"""
    def __init__(self, optimizer) -> None:
        """
        初始化调度器
        
        参数:
            optimizer: 优化器对象，需要包含init_lr属性(初始学习率)
        """
        self.optimizer = optimizer  # 关联的优化器
        self.step_count = 0        # 记录当前步数
    
    @abstractmethod
    def step():
        """抽象方法，子类需实现具体的学习率调整逻辑"""
        pass


class StepLR(scheduler):
    """固定步长学习率衰减策略"""
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        """
        初始化固定步长衰减调度器
        
        参数:
            optimizer: 优化器对象
            step_size: 学习率衰减的步长间隔，默认30步
            gamma: 学习率衰减系数，默认0.1
        """
        super().__init__(optimizer)
        self.step_size = step_size  # 衰减步长间隔
        self.gamma = gamma         # 衰减系数

    def step(self) -> None:
        """执行一步学习率调整"""
        self.step_count += 1
        # 当步数达到衰减间隔时，衰减学习率并重置计数器
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0


class MultiStepLR(scheduler):
    """多阶段步长学习率衰减策略"""
    def __init__(self, optimizer, milestones=[30, 60, 90], gamma=0.1) -> None:
        """
        初始化多阶段步长衰减调度器
        
        参数:
            optimizer: 优化器对象
            milestones: 学习率衰减的里程碑步数列表，默认[30,60,90]
            gamma: 学习率衰减系数，默认0.1
        """
        super().__init__(optimizer)
        self.milestones = milestones          # 衰减里程碑步数列表
        self.gamma = gamma                   # 衰减系数
        self.current_milestone_idx = 0       # 当前里程碑索引

    def step(self) -> None:
        """执行一步学习率调整"""
        self.step_count += 1
        # 检查是否达到当前里程碑步数
        if (self.current_milestone_idx < len(self.milestones) and 
            self.step_count >= self.milestones[self.current_milestone_idx]):
            self.optimizer.init_lr *= self.gamma  # 衰减学习率
            self.current_milestone_idx += 1       # 移动到下一个里程碑


class ExponentialLR(scheduler):
    """指数衰减学习率策略"""
    def __init__(self, optimizer, gamma=0.95) -> None:
        """
        初始化指数衰减调度器
        
        参数:
            optimizer: 优化器对象
            gamma: 每步衰减系数，默认0.95
        """
        super().__init__(optimizer)
        self.gamma = gamma  # 指数衰减系数

    def step(self) -> None:
        """执行一步学习率调整"""
        self.step_count += 1
        # 每步都按指数系数衰减学习率
        self.optimizer.init_lr *= self.gamma
