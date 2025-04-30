import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    改进的训练运行器类，每个epoch计算一次损失和指标，而不是每次迭代都计算
    Modified to calculate loss and metrics once per epoch instead of every iteration
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        """
        初始化训练运行器
        
        参数:
            model: 待训练的模型
            optimizer: 优化器
            metric: 评估指标函数
            loss_fn: 损失函数
            batch_size: 批处理大小，默认为32
            scheduler: 学习率调度器，可选
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric  # 评估指标计算函数
        self.scheduler = scheduler  # 学习率调度器
        self.batch_size = batch_size  # 批处理大小

        # 训练过程中的指标记录
        self.train_scores = []  # 训练集评分记录
        self.dev_scores = []   # 验证集评分记录
        self.train_loss = []    # 训练集损失记录
        self.dev_loss = []     # 验证集损失记录

    def train(self, train_set, dev_set, **kwargs):
        """
        训练模型的主方法
        
        参数:
            train_set: 训练数据集 (X, y)
            dev_set: 验证数据集 (X, y)
            **kwargs: 其他可选参数
                num_epochs: 训练轮数
                log_epochs: 日志打印间隔
                save_dir: 模型保存目录
                patience: 早停耐心值(连续多少轮不提升后停止)
        """
        num_epochs = kwargs.get("num_epochs", 0)  # 训练总轮数
        log_epochs = kwargs.get("log_epochs", 1)  # 日志打印间隔
        save_dir = kwargs.get("save_dir", "best_model")  # 模型保存目录
        patience = kwargs.get("patience", 10)  # 早停机制参数

        # 创建模型保存目录
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0  # 记录最佳验证分数
        no_improve_count = 0  # 记录验证分数未提升的连续轮数

        # 开始训练循环
        for epoch in range(num_epochs):
            # 训练阶段 =============================================
            X, y = train_set
            # 打乱训练数据顺序
            idx = np.random.permutation(range(X.shape[0]))
            X = X[idx]
            y = y[idx]

            total_train_loss = 0.0  # 累计训练损失
            total_train_score = 0.0 # 累计训练评分
            num_batches = 0         # 批次数统计

            n_samples = X.shape[0]  # 总样本数
            # 计算总迭代次数(向上取整)
            n_iterations = (n_samples + self.batch_size - 1) // self.batch_size

            # 批次训练循环
            for iteration in range(n_iterations):
                # 获取当前批次的起止索引
                start = iteration * self.batch_size
                end = start + self.batch_size
                if end > n_samples:
                    end = n_samples
                
                # 获取当前批次数据
                batch_X = X[start:end]
                batch_y = y[start:end]

                # 前向传播
                logits = self.model(batch_X)
                batch_loss = self.loss_fn(logits, batch_y)  # 计算批次损失
                batch_score = self.metric(logits, batch_y)  # 计算批次评分

                # 累计指标
                total_train_loss += batch_loss
                total_train_score += batch_score
                num_batches += 1

                # 反向传播和优化
                self.loss_fn.backward()  # 计算梯度
                self.optimizer.step()    # 更新参数
                if self.scheduler is not None:
                    self.scheduler.step()  # 更新学习率

            # 计算并记录本轮平均训练指标
            avg_train_loss = total_train_loss / num_batches
            avg_train_score = total_train_score / num_batches
            self.train_loss.append(avg_train_loss)
            self.train_scores.append(avg_train_score)

            # 验证阶段 =============================================
            dev_score, dev_loss = self.evaluate(dev_set)
            self.dev_scores.append(dev_score)
            self.dev_loss.append(dev_loss)

            # 早停机制和模型保存逻辑 ================================
            if dev_score > best_score:
                # 验证分数提升，保存最佳模型
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                best_score = dev_score
                no_improve_count = 0  # 重置未提升计数器
            else:
                no_improve_count += 1
                # 检查是否触发早停
                if no_improve_count >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}, no improvement for {patience} consecutive epochs.")
                    break  # 提前终止训练

            # 日志打印 ============================================
            if (epoch + 1) % log_epochs == 0:
                print(f"Epoch: {epoch + 1}/{num_epochs}")
                print(f"[Train] Loss: {avg_train_loss:.4f}, Score: {avg_train_score:.4f}")
                print(f"[Dev]   Loss: {dev_loss:.4f}, Score: {dev_score:.4f}\n")

            # 再次检查早停条件(冗余检查确保退出)
            if no_improve_count >= patience:
                break

        # 记录最佳验证分数
        self.best_score = best_score

    def evaluate(self, data_set):
        """
        在给定数据集上评估模型
        
        参数:
            data_set: 评估数据集 (X, y)
            
        返回:
            score: 模型评分
            loss: 模型损失
        """
        X, y = data_set
        logits = self.model(X)  # 前向传播
        loss = self.loss_fn(logits, y)  # 计算损失
        score = self.metric(logits, y)  # 计算评分
        return score, loss
    
    def save_model(self, save_path):
        """
        保存模型到指定路径
        
        参数:
            save_path: 模型保存路径
        """
        self.model.save_model(save_path)
