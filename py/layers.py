import numpy as np

from tensor import *

class Module:
    def __init__(self): pass 
    def __call__(self, x): return self.forward(x)
    
    def parameters(self):
        params = []
        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                params += param.parameters()
            elif isinstance(param, Tensor):
                if param.requires_grad:
                    params.append(param)
        return params 

    def train(self):
        self.mode = "train"
        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                param.train()

    def eval(self):
        self.mode = "eval"
        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                param.eval()

class Parameter(Tensor):
    def __init__(self, tensor):
        super().__init__(tensor, requires_grad=True)

class Linear(Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.w = Tensor.randn((out_dim, in_dim), requires_grad=True) / np.sqrt(in_dim + out_dim)
        self.b = Tensor.zeros((out_dim, 1), requires_grad=True) if bias else None
        self.bias = bias 

    def forward(self, x):
        out = x @ self.w.T
        if self.bias:
            out += self.b.data.reshape(1, -1)
        return out

class ReLU(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.relu()

class LayerNorm(Module):
    def __init__(self, n_embed):
        super().__init__()
        self.gamma = Tensor.ones((1, n_embed), requires_grad=True)
        self.beta = Tensor.zeros((1, n_embed), requires_grad=True)

    def forward(self, x):
        var_x = x.var(dim=-1, keepdim=True) # (B, T)
        norm_x = (x - x.mean(dim=-1, keepdim=True)) / var_x.sqrt() # (B, T, D)
        z = norm_x * self.gamma + self.beta # (B, T, D)
        return z

class Embedding(Module):
    def __init__(self, in_dim, embed_size):
        super().__init__()
        self.embedding_table = Tensor.randn((in_dim, embed_size), requires_grad=True) / np.sqrt(in_dim)

    def forward(self, idx):
        x = self.embedding_table[idx]
        return x

class Dropout(Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.p = drop_prob 
        self.mode = "train"

    def forward(self, x):
        if self.mode == "eval":
            return x 
        mask = Tensor.rand_like(x) > self.p 
        a = x.masked_fill(mask, 0)
        a /= (1 - self.p)
        return a

class Softmax(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, dim=-1):
        return self.forward(x, dim)

    def forward(self, x, dim=-1):
        x = x.exp()
        out = x / x.sum(dim=dim, keepdim=True)
        return out

class MeanSquaredError(Module):
    def __init__(self): super().__init__()
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        out = (y_pred - y_true)**2 
        out = out.mean()
        return out

class Optim:
    def __init__(self): pass 
    def step(self):
        return NotImplementedError("`step` method not implemented.")
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

class SGD(Optim):
    def __init__(self, params, lr=1e-3, reg=0):
        self.params = params 
        self.lr = lr 
        self.reg = reg 

    def step(self):
        for param in self.params:
            param.data -= self.lr * (param.grad.data + self.reg * param.data)

