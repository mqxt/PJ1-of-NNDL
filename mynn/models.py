from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_rates=None):
        self.size_list = size_list
        self.act_func = act_func
        self.lambda_list = lambda_list
        self.dropout_rates = dropout_rates

        if size_list is not None and act_func is not None:
            self.layers = []
            num_hidden_layers = len(size_list)-2
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                self.layers.append(layer)

                # 非输出层添加激活函数和Dropout
                if i < len(size_list) - 2:
                    if act_func == 'Logistic':
                        raise NotImplementedError
                    elif act_func == 'ReLU':
                        self.layers.append(ReLU())
                    if dropout_rates is not None and i < len(dropout_rates):
                        self.layers.append(Dropout(p=dropout_rates[i]))

    def train_mode(self):
        for layer in self.layers:
            layer.train()

    def eval_mode(self):
        for layer in self.layers:
            layer.eval()

            
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def compute_l2_loss(self):
        """
        Compute the total L2 regularization loss for all layers
        """
        l2_loss = 0.0
        for layer in self.layers:
            if isinstance(layer, Linear) and layer.weight_decay: 
                lambda_val = layer.weight_decay_lambda
                l2_loss += 0.5 * lambda_val * np.sum(layer.params['W'] ** 2)
        return l2_loss

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        
        # 重建模型结构（包括 Dropout）
        self.layers = []
        num_hidden_layers = len(self.size_list) - 2
        for i in range(len(self.size_list) - 1):
            # 加载线性层
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i + 2]['lambda']
            self.layers.append(layer)
            
            
            if i < len(self.size_list) - 2:
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    self.layers.append(ReLU())
            
            
            if self.dropout_rates is not None and i < len(self.dropout_rates):
                self.layers.append(Dropout(p=self.dropout_rates[i]))
            
    
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):  # 只保存可优化层（Linear）
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
            # 如果需要保存 Dropout 的 p 值，可以额外添加逻辑
            # elif isinstance(layer, Dropout):
            #     param_list.append({'p': layer.p})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
