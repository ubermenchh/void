import numpy as np

# Helper Functions
def is_tensor(obj): return isinstance(obj, Tensor)
def to_tensor(obj): return Tensor(obj) if not is_tensor(obj) else obj


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
        self._grad_fn = None 
        self._ctx = None

    def __repr__(self):
        rounded_data = np.around(self.data, decimals=5)
        return f"""Tensor({rounded_data}, requires_grad={self.requires_grad})"""

    def set_grad_fn(self, grad_fn):
        self._grad_fn = grad_fn 

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        if self.grad is None:
            self.grad = grad 
        else:
            self.grad += grad 
        if self._grad_fn:
            grads = self._grad_fn(self._ctx, grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            for t, g in zip(self._ctx.saved_tensors, grads):
                if t.requires_grad:
                    t.backward(g)
    
    def shape(self):
        return self.data.shape

    def sum(self, dim=-1, keepdim=False): return Sum.apply(self, dim, keepdim)

    def __neg__(self):        return Neg.apply(self)
    def __add__(self, other): return Add.apply(self, to_tensor(other))
    def __sub__(self, other): return Add.apply(self, -to_tensor(other)) 
    def __mul__(self, other): return Mul.apply(self, to_tensor(other))
    def __truediv__(self, other): return Div.apply(self, to_tensor(other))


# Tensor Operations
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

class Mul(Function):
    @staticmethod 
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b) 
        result = Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)
        return result 

    @staticmethod 
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors 
        grad_a = grad_output * b.data if a.requires_grad else None 
        grad_b = grad_output * a.data if b.requires_grad else None 
        return grad_a, grad_b

class Div(Function):
    @staticmethod 
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b) 
        result = Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad)
        return result 

    @staticmethod 
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors 
        grad_a = grad_output / b.data if a.requires_grad else None 
        grad_b = -grad_output * a.data / (b.data**2) if b.requires_grad else None 
        return grad_a, grad_b

class Sum(Function):
    @staticmethod 
    def forward(ctx, a, dim, keepdim):
        ctx.save_for_backward(a)
        _data = a.data.sum(axis=dim, keepdims=keepdim)
        return Tensor(_data, requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        if a.requires_grad:
            grad_a = np.ones_like(a.data) * grad_output
            return grad_a 
        return None
