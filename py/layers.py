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


class Linear(Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.w = randn((in_dim, out_dim), requires_grad=True) / np.sqrt(in_dim)
        self.b = zeros(out_dim, requires_grad=True)
        self.bias = bias 

    def forward(self, x):
        out = x @ self.w 
        if self.bias:
            out += self.b 
        return out
