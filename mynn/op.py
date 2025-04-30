from abc import abstractmethod
import numpy as np

class Layer():
    """
    神经网络层的基类，定义了所有层共有的接口和属性
    """
    def __init__(self) -> None:
        self.optimizable = True  # 标记该层是否包含可优化参数

    def train(self):
        """设置层为训练模式"""
        self.training = True

    def eval(self):
        """设置层为评估模式"""
        self.training = False
    
    @abstractmethod
    def forward():
        """前向传播抽象方法，必须由子类实现"""
        pass

    @abstractmethod
    def backward():
        """反向传播抽象方法，必须由子类实现"""
        pass


class Linear(Layer):
    """
    全连接层(线性层)实现
    需要实现前向传播(forward)和反向传播(backward)方法
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal) -> None:
        """
        初始化全连接层
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param initialize_method: 权重初始化方法
        """
        super().__init__()
        # 使用He初始化方法初始化权重，有助于缓解深度网络中的梯度消失/爆炸问题
        self.b = np.zeros((1, out_dim))  # 偏置初始化为0
        self.W = np.random.normal(0, np.sqrt(2 / in_dim), size=(in_dim, out_dim))

        self.grads = {'W' : None, 'b' : None}  # 存储权重和偏置的梯度
        self.input = None  # 记录前向传播的输入，用于反向传播

        self.params = {'W' : self.W, 'b' : self.b}  # 所有可训练参数

            
    
    def __call__(self, X) -> np.ndarray:
        """使层实例可以像函数一样调用"""
        return self.forward(X)

    def forward(self, X):
        """
        前向传播计算
        :param X: 输入数据，形状为[batch_size, in_dim]
        :return: 输出数据，形状为[batch_size, out_dim]
        """
        self.input = X  # 保存输入用于反向传播
        output = np.matmul(X, self.W) + self.b  # 线性变换: WX + b
        return output

    def backward(self, grad : np.ndarray):
        """
        反向传播计算
        :param grad: 来自下一层的梯度，形状为[batch_size, out_dim]
        :return: 传递给前一层的梯度，形状为[batch_size, in_dim]
        此函数同时计算W和b的梯度
        """
        batch_size = self.input.shape[0]
        # 计算权重梯度: dL/dW = X^T * dL/dY / batch_size
        self.grads['W'] = np.matmul(self.input.T, grad) / batch_size
        # 计算偏置梯度: dL/db = sum(dL/dY) / batch_size
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size
        
        # 计算传递给前一层的梯度: dL/dX = dL/dY * W^T
        d_input = np.matmul(grad, self.W.T)
        
        return d_input
    
    def clear_grad(self):
        """清空梯度"""
        self.grads = {'W' : None, 'b' : None}

class ReLU(Layer):
    """
    ReLU激活函数层
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None  # 保存输入值用于反向传播
        self.optimizable = False  # 标记该层没有可优化参数

    def __call__(self, X):
        # 使实例可以像函数一样调用
        return self.forward(X)

    def forward(self, X):
        """
        前向传播计算
        参数:
            X: 输入数据
        返回:
            输出数据，负值置为0
        """
        self.input = X  # 保存输入用于反向传播
        output = np.where(X<0, 0, X)  # ReLU计算：小于0的值设为0
        return output
    
    def backward(self, grads):
        """
        反向传播计算
        参数:
            grads: 上一层传来的梯度
        返回:
            传递给下一层的梯度
        """
        assert self.input.shape == grads.shape  # 确保输入和梯度形状一致
        # 计算梯度：输入小于0的位置梯度为0，其他位置保持原梯度
        output = np.where(self.input < 0, 0, grads)
        return output
        
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

class MultiCrossEntropyLoss(Layer):
    """
    多分类交叉熵损失层，内部包含Softmax层(可通过cancel_softmax取消)
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.optimizable = False  # 标记该层没有可优化参数
        self.model = model  # 关联的模型
        self.max_classes = max_classes  # 最大类别数
        self.has_softmax = True  # 是否包含softmax
        self.logits = None  # 原始预测值
        self.probs = None  # 概率预测值
        self.labels = None  # 真实标签
        self.grads = None  # 梯度
        self.batch_size = 0  # 批大小

    def __call__(self, predicts, labels):
        # 使实例可以像函数一样调用
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        前向传播计算损失
        参数:
            predicts: [batch_size, D] 预测值
            labels: [batch_size,] 真实标签
        返回:
            交叉熵损失值
        """
        self.logits = predicts
        self.labels = labels
        self.batch_size = predicts.shape[0]
        
        # 如果需要，应用softmax
        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts
        
        # 创建one-hot编码的标签
        y_one_hot = np.zeros((self.batch_size, self.max_classes))
        y_one_hot[np.arange(self.batch_size), labels] = 1
        
        # 计算交叉熵损失(添加小量防止log(0))
        loss = -np.sum(y_one_hot * np.log(self.probs + 1e-10)) / self.batch_size
        
        return loss
    
    def backward(self):
        """
        反向传播计算梯度
        """
        # 创建one-hot编码的标签
        y_one_hot = np.zeros((self.batch_size, self.max_classes))
        y_one_hot[np.arange(self.batch_size), self.labels] = 1
        
        if self.has_softmax:
            # 如果包含softmax，梯度为(softmax输出 - one-hot标签)
            self.grads = (self.probs - y_one_hot) / self.batch_size
        else:
            # 如果不包含softmax，直接计算梯度
            self.grads = -y_one_hot / (self.probs + 1e-10) / self.batch_size
        
        # 将梯度传递给模型进行反向传播
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        """
        取消内部的softmax计算
        """
        self.has_softmax = False
        return self
    
