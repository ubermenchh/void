import numpy as np

# Helper Functions
def is_tensor(obj): return isinstance(obj, Tensor)
def to_tensor(obj): return Tensor(obj) if not is_tensor(obj) else obj
def rand_tensor(shape, requires_grad=False):
    return Tensor(np.random.rand(*shape), requires_grad=requires_grad)
def randn_tensor(shape, requires_grad=False):
    return Tensor(np.random.randn(*shape), requires_grad=requires_grad)

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
        return f"""Tensor(\n{rounded_data}, requires_grad={self.requires_grad}\n)"""

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


    def sum(self, dim=None, keepdim=False): return Sum.apply(self, dim, keepdim)
    def mean(self, dim=None, keepdim=False): return Mean.apply(self, dim, keepdim)
    
    def relu(self): return Relu.apply(self)

    def __neg__(self):        return Neg.apply(self)
    def __add__(self, other): return Add.apply(self, to_tensor(other))
    def __sub__(self, other): return Add.apply(self, -to_tensor(other)) 
    def __mul__(self, other): return Mul.apply(self, to_tensor(other))
    def __truediv__(self, other): return Div.apply(self, to_tensor(other))
    def __pow__(self, other): return Pow.apply(self, to_tensor(other))
    def __matmul__(self, other): return Matmul.apply(self, to_tensor(other))

    def __radd__(self, other): return Add.apply(self, to_tensor(other))
    def __rsub__(self, other): return Add.apply(self, -to_tensor(other)) 
    def __rmul__(self, other): return Mul.apply(self, to_tensor(other))
    def __rtruediv__(self, other): return Div.apply(self, to_tensor(other))
    def __rpow__(self, other): return Pow.apply(self, to_tensor(other))
 
    def log(self): return Log.apply(self)
    def sqrt(self): return Sqrt.apply(self)
    def sin(self): return Sin.apply(self)
    def cos(self): return Cos.apply(self)
    def exp(self): return Exp.apply(self)
    def tan(self): return self.sin() / self.cos()

    def transpose(self, axes=None): return Transpose.apply(self, axes)
    @property 
    def T(self): return self.transpose()

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

class Pow(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data**b.data, requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        if a.requires_grad:
            grad_a = b.data * (a.data ** (b.data - 1)) * grad_output
            return grad_a 
        return None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.log(a.data), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        if a.requires_grad:
            grad_a = (1. / a.data) * grad_output
            return grad_a 
        return None

class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.sqrt(a), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        if a.requires_grad:
            grad_a = (1. / 2 * np.sqrt(a.data)) * grad_output
            return grad_a 
        return None

class Sin(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.sin(a.data), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        if a.requires_grad:
            grad_a = np.cos(a.data) * grad_output
            return grad_a 
        return None

class Cos(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.cos(a.data), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        if a.requires_grad:
            grad_a = -np.sin(a.data) * grad_output
            return grad_a 
        return None

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.exp(a.data), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        if a.requires_grad:
            grad_a = np.exp(a.data) * grad_output
            return grad_a 
        return None

class Mean(Function): 
    @staticmethod 
    def forward(ctx, a, dim, keepdim):
        ctx.save_for_backward(a, dim, keepdim)
        return Tensor(a.data.mean(axis=dim, keepdims=keepdim), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, dim, keepdim = ctx.saved_tensors  

        if a.requires_grad:
            if dim is None:
                n = np.prod(a.data.shape)
            else:
                n = a.data.shape[dim]
            grad_a = np.ones_like(a.data) * grad_output / n 
            if not keepdim and dim is not None:
                grad_a = np.expand_dims(grad_a, axis=dim)
            return grad_a
        return None

class Relu(Function):
    @staticmethod 
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.maximum(a.data, 0), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors 
        if a.requires_grad:
            grad_a = (a.data > 0) * grad_output 
            return grad_a 
        return None

class Matmul(Function):
    @staticmethod 
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(np.matmul(a.data, b.data), requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output @ b.data.T if a.requires_grad else None 
        grad_b = a.data.T @ grad_output if b.requires_grad else None 
        return grad_a, grad_b

class Transpose(Function):
    @staticmethod 
    def forward(ctx, a, *axes):
        ctx.save_for_backward(a, axes)
        return Tensor(np.transpose(a.data, *axes), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, axes = ctx.saved_tensors 
        if a.requires_grad:
            grad_a = np.transpose(grad_output, *axes)
            return grad_a 
        return None
