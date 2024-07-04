import numpy as np

# Helper Functions
def is_tensor(obj): return isinstance(obj, Tensor)
def to_tensor(obj): return Tensor(obj) if not is_tensor(obj) else obj


class Tensor:
    _compute_grad = True

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad 
        
        self.grad = None 
        self._ctx = None

    def __repr__(self):
        rounded_data = np.around(self.data, decimals=5)
        return f"""Tensor(\n{rounded_data}, requires_grad={self.requires_grad}\n)"""

    def __getitem__(self, idx): return Slice.apply(self, idx)
    def __setitem__(self, idx, value): self.data[idx] = value
    def __hash__(self): return id(self)
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def numpy(self): return self.data.copy()

    def backward(self):
        if self._ctx is None: return
        if self.size != 1:
            raise ValueError(f"backward can only be called for scalar tensors, but got of shape {self.shape}")
        if self.grad is None: self.grad = Tensor(1.0)

        def build_topo(tensor, visited, nodes):
            if tensor not in visited:
                visited.add(tensor)
                if tensor._ctx:
                    for parent in tensor._ctx.saved_tensors:
                        if parent not in visited: build_topo(parent, visited, nodes)
                nodes.append(tensor)
            return nodes
        
        nodes = build_topo(self, set(), [])

        for tensor in reversed(nodes):
            if tensor._ctx:
                grads = tensor._ctx.backward(tensor._ctx, tensor.grad)
                for parent, grad in zip(tensor._ctx.saved_tensors, grads):
                    if grad is not None and parent.requires_grad:
                        assert grad.shape == parent.shape, f"grad shape must match tensor shape, {grad.shape} != {parent.shape}"
                        parent.grad = grad if parent.grad is None else (parent.grad + grad)
        return self

    def repeat(self, *sizes):
        new_data = np.tile(self.data, sizes)
        return Tensor(new_data, requires_grad=self.requires_grad)

    @staticmethod 
    def randn(shape, requires_grad=False, **kwargs):
        return Tensor(np.random.randn(*shape, **kwargs), requires_grad=requires_grad)
    @staticmethod 
    def rand(shape, requires_grad=False, **kwargs):
        return Tensor(np.random.rand(*shape, **kwargs), requires_grad=requires_grad)
    @staticmethod
    def zeros(shape, requires_grad=False, **kwargs):
        return Tensor(np.zeros(shape, **kwargs), requires_grad=requires_grad)
    @staticmethod
    def ones(shape, requires_grad=False, **kwargs):
        return Tensor(np.ones(shape, **kwargs), requires_grad=requires_grad)
    @staticmethod 
    def eye(size, requires_grad=False, **kwargs):
        return Tensor(np.eye(size, **kwargs), requires_grad=requires_grad)
    @staticmethod
    def randn_like(tensor, requires_grad=False, **kwargs):
        assert(isinstance(tensor, Tensor))
        return Tensor(np.random.randn(*tensor.shape, **kwargs), requires_grad=requires_grad)
    @staticmethod 
    def rand_like(tensor, requires_grad=False, **kwargs):
        assert(isinstance(tensor, Tensor))
        return Tensor(np.random.rand(*tensor.shape, **kwargs), requires_grad=requires_grad)
    @staticmethod 
    def zeros_like(tensor, requires_grad=False, **kwargs):
        assert(isinstance(tensor, Tensor))
        return Tensor(np.zeros_like(tensor.data, **kwargs), requires_grad=requires_grad)
    @staticmethod 
    def ones_like(tensor, requires_grad=False, **kwargs):
        assert(isinstance(tensor, Tensor))
        return Tensor(np.ones_like(tensor.data, **kwargs), requires_grad=requires_grad)
    @staticmethod 
    def full(shape, fill_value, requires_grad=False, **kwargs):
        return Tensor(np.full(shape, fill_value, **kwargs), requires_grad=requires_grad)
    @staticmethod 
    def full_like(tensor, fill_value, requires_grad=False, **kwargs):
        assert(isinstance(tensor, Tensor))
        return Tensor(np.full_like(tensor.data, fill_value, **kwargs), requires_grad=requires_grad)
    @staticmethod 
    def tri(size, requires_grad=False, **kwargs):
        return Tensor(np.tri(N=size, **kwargs), requires_grad=requires_grad)

    def triu(self, diag=0, **kwargs): return Tensor(np.triu(self.data, k=diag, **kwargs), requires_grad=self.requires_grad)
    def tril(self, diag=0, **kwargs): return Tensor(np.tril(self.data, k=diag, **kwargs), requires_grad=self.requires_grad)


    def sum(self, dim=None, keepdim=False):  return Sum.apply(self, dim, keepdim)
    def max(self, dim=None, keepdim=False):  return Max.apply(self, dim, keepdim)
    def min(self, dim=None, keepdim=False):  return Min.apply(self, dim, keepdim)
    def mean(self, dim=None, keepdim=False): return Mean.apply(self, dim, keepdim)
    def var(self, dim=None, keepdim=False):  return Var.apply(self, dim, keepdim)
    def std(self, dim=None, keepdim=False):  return self.var(dim, keepdim).sqrt()
    
    def add(self, other):           return Add.apply(self, to_tensor(other))
    def sub(self, other):           return Sub.apply(self, to_tensor(other))
    def multiply(self, other):      return Mul.apply(self, to_tensor(other))
    def divide(self, other):        return Div.apply(self, to_tensor(other))
    def pow(self, other):           return Pow.apply(self, to_tensor(other))
    def matmul(self, other):        return Matmul.apply(self, to_tensor(other))

    def __neg__(self):              return Neg.apply(self)
    def __add__(self, other):       return Add.apply(self, to_tensor(other))
    def __sub__(self, other):       return Sub.apply(self, to_tensor(other))
    def __mul__(self, other):       return Mul.apply(self, to_tensor(other))
    def __truediv__(self, other):   return Div.apply(self, to_tensor(other))
    def __pow__(self, other):       return Pow.apply(self, to_tensor(other))
    def __matmul__(self, other):    return Matmul.apply(self, to_tensor(other))

    def __radd__(self, other):      return Add.apply(to_tensor(other), self)
    def __rsub__(self, other):      return Sub.apply(to_tensor(other), self)
    def __rmul__(self, other):      return Mul.apply(to_tensor(other), self)
    def __rtruediv__(self, other):  return Div.apply(to_tensor(other), self)
    def __rpow__(self, other):      return Pow.apply(to_tensor(other), self)
    def __rmatmul__(self, other):   return Matmul(to_tensor(other), self)

    def __iadd__(self, other):      return Add.apply(self, to_tensor(other))
    def __isub__(self, other):      return Sub.apply(self, to_tensor(other))
    def __imul__(self, other):      return Mul.apply(self, to_tensor(other))
    def __itruediv__(self, other):  return Div.apply(self, to_tensor(other))

    def __lt__(self, other): return self.data < other.data
    def __gt__(self, other): return self.data > other.data 
    def __ge__(self, other): return self.data >= other.data 
    def __le__(self, other): return self.data <= other.data 
    def __ne__(self, other): return self.data != other.data 
    def __eq__(self, other): return self.data == other.data

    def log(self): return Log.apply(self)
    def sqrt(self): return Sqrt.apply(self)
    def sin(self): return Sin.apply(self)
    def cos(self): return Cos.apply(self)
    def exp(self): return Exp.apply(self)
    def tan(self): return self.sin() / self.cos()

    def relu(self): return Relu.apply(self)
    
    @property
    def shape(self): return self.data.shape
    @property 
    def dtype(self): return self.data.dtype
    @property
    def size(self): return self.data.size
    @property 
    def ndim(self): return self.data.ndim

    def transpose(self, axes=None): return Transpose.apply(self, axes)
    @property 
    def T(self): return self.transpose()
    def detach(self): self.requires_grad = False
    def reshape(self, *shape): return Reshape.apply(self, *shape)
    def concat(self, others, dim=0):
        if not isinstance(others, (list, tuple)):
            others = [others]
        return Concat.apply([self] + others, dim)
    def stack(self, others, dim=0):
        if not isinstance(others, (list, tuple)):
            others = [others]
        return Stack.apply([self] + others, dim)
    def masked_fill(self, condition, value):
        return MaskedFill.apply(self, condition, value)
    def repeat(self, shape):
        return Tensor.ones(shape) * self

    def cross_entropy(self, other): return CELoss.apply(self, to_tensor(other))

def arange(start, stop, step=None, **kwargs):
    return Tensor(np.arange(start, stop, step, **kwargs))
def exp(tensor): return Tensor.exp(tensor)
def log(tensor): return Tensor.log(tensor)
def max(tensor, dim=None, keepdim=None): return Tensor.max(tensor, dim=dim, keepdim=keepdim)
def min(tensor, dim=None, keepdim=None): return Tensor.min(tensor, dim=dim, keepdim=keepdim)
def sum(tensor, dim=None, keepdim=None): return Tensor.sum(tensor, dim=dim, keepdim=keepdim)
def mean(tensor, dim=None, keepdim=None): return Tensor.mean(tensor, dim=dim, keepdim=keepdim)

# Tensor Operations
class Function:
    def __init__(self):
        self.save_tensor = ()

    @classmethod 
    def apply(cls, *args):
        ctx = cls()
        result = cls.forward(ctx, *args)
        if any(isinstance(t, Tensor) and t.requires_grad for t in args):
            result.requires_grad = True
            result._ctx = ctx  
        return result 

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
   
    @staticmethod
    def forward(ctx, *args): raise NotImplementedError 
    @staticmethod
    def backward(ctx, grad): raise NotImplementedError 

class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(-a.data, requires_grad=a.requires_grad)
    
    @staticmethod
    def backward(ctx, grad):
        return -grad, None

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors 
        return grad if a.requires_grad else None, grad if b.requires_grad else None

class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    @staticmethod
    def backward(ctx, grad):
        return grad, -grad

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        grad_a = grad * b.data
        grad_b = grad * a.data
        return grad_a, grad_b

class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        grad_a = grad / b.data 
        grad_b = -grad * a.data / (b.data**2) 
        return grad_a, grad_b

class Sum(Function):
    @staticmethod
    def forward(ctx, x, dim, keepdim):
        ctx.save_for_backward(x)
        ctx.dim, ctx.keepdim = dim, keepdim
        return Tensor(x.data.sum(axis=dim, keepdims=keepdim))

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        out = np.broadcast_to(grad.data, x.shape)
        return out, None

class Max(Function):
    @staticmethod
    def forward(ctx, a, dim, keepdim):
        ctx.save_for_backward(a)
        ctx.dim, ctx.keepdim = dim, keepdim
        _data = a.data.max(axis=dim, keepdims=keepdim)
        return Tensor(_data, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors
        dim, keepdim = ctx.dim, ctx.keepdim
        if not a.requires_grad:
            return None, None 
        
        mask = (a.data == np.max(a.data, axis=dim, keepdims=keepdim))
        if not keepdim and dim is not None:
            grad.data = np.expand_dims(grad.data, axis=dim)

        grad_a = mask * grad.data
        if not keepdim and dim is not None:
            grad_a = np.sum(grad_a, axis=dim, keepdims=keepdim)
        return grad_a, None

class Min(Function):
    @staticmethod
    def forward(ctx, a, dim, keepdim):
        ctx.save_for_backward(a)
        ctx.dim, ctx.keepdim = dim, keepdim
        _data = a.data.min(axis=dim, keepdims=keepdim)
        return Tensor(_data, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        dim, keepdim = ctx.dim, ctx.keepdim
        if not a.requires_grad:
            return None, None
        
        mask = (a.data == np.min(a.data, axis=dim, keepdims=keepdim))
        if not keepdim and dim is not None:
            grad_output = np.expand_dims(grad_output, axis=dim)

        grad_a = mask * grad_output
        if not keepdim and dim is not None:
            grad_a = np.sum(grad_a, axis=dim, keepdims=keepdim)
        return grad_a, None

class Pow(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data**b.data, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        if a.requires_grad:
            grad_a = b.data * (a.data ** (b.data - 1)) * grad.data
            return grad_a, None
        return None, None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.log(a.data), requires_grad=a.requires_grad)
    
    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors
        if a.requires_grad:
            grad_a = (1. / a.data) * grad.data
            return grad_a, None
        return None, None

class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.sqrt(a.data), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors
        if a.requires_grad:
            grad_a = (1. / 2 * a.data) * grad.data
            return grad_a, None 
        return None, None

class Sin(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.sin(a.data), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors
        if a.requires_grad:
            grad_a = np.cos(a.data) * grad.data
            return grad_a, None
        return None, None

class Cos(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.cos(a.data), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors
        if a.requires_grad:
            grad_a = -np.sin(a.data) * grad.data
            return grad_a, None
        return None, None

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.exp(a.data), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors
        if a.requires_grad:
            grad_a = np.exp(a.data) * grad.data
            return grad_a, None 
        return None, None

class Mean(Function):
    @staticmethod
    def forward(ctx, a, dim, keepdim):
        ctx.save_for_backward(a)
        ctx.input_shape = a.data.shape

        if dim is None:
            ctx.num_el = a.data.size 
        elif isinstance(dim, int):
            ctx.num_el = a.data.shape[dim]
        else:
            ctx.num_el = np.prod([a.data.shape[d] for d in dim])

        return Tensor(a.data.mean(axis=dim, keepdims=keepdim), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        num_el, input_shape = ctx.num_el, ctx.input_shape
        
        grad_input = np.broadcast_to(grad.data / num_el, input_shape)
        return grad_input, None

class Var(Function):
    @staticmethod
    def forward(ctx, a, dim, keepdim):
        ctx.save_for_backward(a)
        ctx.dim = dim
        return Tensor(a.data.var(axis=dim, keepdims=keepdim), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors
        dim = ctx.dim

        if a.requires_grad:
            grad_a = np.ones_like(a.data) * grad.data 
            grad_a = grad_a * 2 * (a.data - a.data.mean(axis=dim, keepdims=True)) / np.prod(np.array(a.shape)[dim])
            return grad_a, None 
        return None, None

class Relu(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.maximum(a.data, 0), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors
        if a.requires_grad:
            grad_a = np.where(a.data > 0, 1, 0) * grad
            return grad_a, None 
        return None, None

class Matmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        grad_a = grad @ b.data.T
        grad_b = a.data.T @ grad

        return grad_a, grad_b

class Transpose(Function):
    @staticmethod
    def forward(ctx, a, *axes):
        ctx.save_for_backward(a)
        ctx.axes = axes
        return Tensor(np.transpose(a.data, *axes), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        axes = ctx.axes
        grad_a = np.transpose(grad, *axes)
        return grad_a, None 

class Slice(Function):
    @staticmethod
    def forward(ctx, a, idx):
        ctx.save_for_backward(a)
        ctx.idx = idx
        return Tensor(a.data[idx], requires_grad=a.requires_grad)
    
    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors
        idx = ctx.idx
        if a.requires_grad:
            grad_a = np.zeros_like(a.data)
            grad_a[idx] = grad.data 
            return grad_a, None 
        return None, None

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, *shape):
        ctx.save_for_backward(a)
        ctx.shape = shape
        return Tensor(a.data.reshape(shape), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors
        shape = ctx.shape
        if a.requires_grad:
            grad_a = grad.data.reshape(shape)
            return grad_a, None
        return None, None

class Concat(Function):
    @staticmethod
    def forward(ctx, tensors, dim=0):
        ctx.save_for_backward(tensors)
        ctx.dim = dim
        new_data = np.concatenate([t.data for t in tensors], axis=dim)
        requires_grad = any(t.requires_grad for t in tensors)
        return Tensor(new_data, requires_grad=requires_grad)
    
    @staticmethod
    def backward(ctx, grad):
        tensors, = ctx.saved_tensors
        dim = ctx.dim
        grad_data = grad.data
        grads = np.split(grad_data, [t.shape[dim] for t in tensors[:-1]], axis=dim)
        return [grads], None

class Stack(Function):
    @staticmethod
    def forward(ctx, tensors, dim=0):
        ctx.save_for_backward(tensors)
        ctx.dim = dim
        new_data = np.stack([t.data for t in tensors], axis=dim)
        requires_grad = any(t.requires_grad for t in tensors)
        return Tensor(new_data, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx, grad):
        dim = ctx.dim
        grad_data = grad.data
        grads = np.split(grad_data, grad_data.shape[dim], axis=dim)
        return tuple([*[grads], None])

class MaskedFill(Function):
    @staticmethod
    def forward(ctx, a, condition, value):
        ctx.save_for_backward(a)
        ctx.condition, ctx.value = condition, value
        data = np.where(condition, a.data, value)
        return Tensor(data, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors
        condition, _ = ctx.condition, ctx.value
        if a.requires_grad:
            grad_a = np.where(condition, grad.data, 0)
            return grad_a, None 
        return None, None

class CELoss(Function): # Cross Entropy Loss
    @staticmethod 
    def forward(ctx, y_pred, y_true):
        ctx.save_for_backward(y_pred, y_true)
        exps = np.exp(y_pred.data - np.max(y_pred.data, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True) + 1e-22
        log_l = -np.log(probs[np.arange(len(y_true.data)), y_true.data.astype(int)])
        return Tensor(log_l.mean())

    @staticmethod 
    def backward(ctx, grad):
        y_pred, y_true = ctx.saved_tensors
        exps = np.exp(y_pred.data - np.max(y_pred.data, axis=1, keepdims=True)) + 1e-22 
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        dloss = np.zeros_like(probs)
        dloss[np.arange(len(y_true.data)), y_true.data.astype(int)] -= 1 
        dloss += probs 
        dloss /= len(y_true.data)
        return [dloss * grad.data, None]
