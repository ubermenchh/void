import numpy as np 

class Context:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors 
       
    @property
    def saved_tensors(self):
        return self._saved_tensors
    
    @saved_tensors.setter
    def saved_tensors(self, tensors):
        self._saved_tensors = tensors

class Function:
    @staticmethod 
    def forward(ctx, *args):
        raise NotImplementedError("forward method not implemented")
    
    @staticmethod
    def backward(ctx, *grad_output):
        raise NotImplementedError("backward method not implemented")

    @classmethod 
    def apply(cls, *args):
        ctx = Context()
        result = cls.forward(ctx, *args)
        result.set_grad_fn(cls.backward)
        result._ctx = ctx 
        return result


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad 
        
        self.grad = None 
        self.grad_fn = None 

    def __repr__(self):
        return f"""Tensor({self.data}, requires_grad={self.requires_grad})"""

    def set_grad_fn(self, grad_fn):
        self.grad_fn = grad_fn 

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        if self.grad is None:
            self.grad = grad 
        else:
            self.grad += grad 
        if self.grad_fn:
            self.grad_fn.backward(grad)

    def __neg__(self):
        return Neg.apply(self)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Add.apply(self, other)
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Add.apply(self, -other)

class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(-a.data, requires_grad=a.requires_grad)
    
    @staticmethod 
    def backward(ctx, grad_output):
        a = ctx.saved_tensors 
        return -grad_output if a.requires_grad else None

class Add(Function):
    @staticmethod 
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        result = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
        return result 

    @staticmethod 
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output if a.requires_grad else None 
        grad_b = grad_output if b.requires_grad else None 
        return grad_a, grad_b



if __name__=="__main__":
    a = Tensor([1, 2, 3, 4], requires_grad=True)
    b = Tensor([4, 5, 7, 2], requires_grad=True)

    c = a - b 
    
    print(a)
    print(b)
    print(c)
