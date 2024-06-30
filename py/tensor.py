import numpy as np

# Helper Functions
def is_tensor(obj): return isinstance(obj, Tensor)
def to_tensor(obj): return Tensor(obj) if not is_tensor(obj) else obj


class Function:
    __slots__ = (
        "op",
        "args",
    )

    def __init__(self, op, *args):
        self.op: Function = op
        self.args: list[Tensor] = args

    @classmethod
    def apply(cls, *args):
        ctx = Function(cls, *args)
        result = cls.forward(*args)
        if Function._is_part_of_graph(ctx):
            result._ctx = ctx
        return result

    @staticmethod
    def _is_part_of_graph(ctx):
        if not Tensor._compute_grad:
            return False

        for node in ctx.args:
            if isinstance(node, Tensor) and (
                node.requires_grad or node._ctx is not None
            ):
                return True
        return False

    @staticmethod
    def forward(self, *args):
        raise NotImplementedError

    @staticmethod
    def backward(self, *args):
        raise NotImplementedError

class Tensor:
    _compute_grad = True

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad 
        
        self.grad = None 
        self._grad_fn = None 
        self._ctx = None

    def __repr__(self):
        rounded_data = np.around(self.data, decimals=5)
        return f"""Tensor(\n{rounded_data}, requires_grad={self.requires_grad}\n)"""

    def __getitem__(self, idx): return Slice.apply(self, idx)
    def __setitem__(self, idx, value): self.data[idx] = value
    def __hash__(self): return id(self)


    def set_grad_fn(self, grad_fn):
        self._grad_fn = grad_fn 
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def _undo_broadcast(self, tensor, grad):
        data = tensor.data
        grad = grad.data

        while len(data.shape) != len(grad.shape):
            grad = grad.sum(axis=0, keepdims=(len(grad.shape) == 1))

        for idx, (s1, s2) in enumerate(zip(data.shape, grad.shape)):
            if s1 < s2:
                grad = grad.sum(axis=idx, keepdims=True)

        return Tensor(grad)

    def numpy(self): return self.data.copy()

    def backward(self):
        if self._ctx is None:
            return

        if self.grad is None:
            if self.size != 1:
                raise RuntimeError("Backward can not be called on non zero tensor")
            self.grad = Tensor([1.0])

        def topo_sort(node: Tensor, visited: set, sortlist: list) -> list:
            if not isinstance(node, Tensor) or node in visited:
                return sortlist
            visited.add(node)
            if node._ctx is None:
                sortlist.append(node)
                return sortlist
            for child_node in node._ctx.args:
                topo_sort(child_node, visited, sortlist)
            sortlist.append(node)
            return sortlist

        node_list: list[Tensor] = reversed(topo_sort(self, set(), []))

        for node in node_list:
            if node._ctx is None:
                continue
            grads = node._ctx.op.backward(node._ctx, node.grad)
            if len(node._ctx.args) == 1:
                grads = [grads]

            for tensor, grad in zip(node._ctx.args, grads):
                if grad is None:
                    continue
                grad = self._undo_broadcast(tensor, grad)
                if tensor.grad is None:
                    tensor.grad = Tensor(np.zeros_like(tensor.data).astype(np.float32))
                tensor.grad.data += grad.numpy()

            node._ctx = None

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


    def sum(self, dim=None, keepdim=False): return Sum.apply(self, dim, keepdim)
    def max(self, dim=None, keepdim=False): return Max.apply(self, dim, keepdim)
    def min(self, dim=None, keepdim=False): return Min.apply(self, dim, keepdim)
    def mean(self, dim=None, keepdim=False): return Mean.apply(self, dim, keepdim)
    def var(self, dim=None, keepdim=False): return Var.apply(self, dim, keepdim)
    def std(self, dim=None, keepdim=False): return self.var(dim, keepdim).sqrt()

    def __neg__(self):              return Neg.apply(self)
    def __add__(self, other):       return Add.apply(self, to_tensor(other))
    def __sub__(self, other):       return Sub.apply(self, to_tensor(other))
    def __mul__(self, other):       return Mul.apply(self, to_tensor(other))
    def __truediv__(self, other):   return Div.apply(self, to_tensor(other))
    def __pow__(self, other):       return Pow.apply(self, to_tensor(other))
    def __matmul__(self, other):    return Matmul.apply(self, to_tensor(other))

    def __radd__(self, other):      return Add.apply(to_tensor(other), self)
    def __rsub__(self, other):      return other + -self
    def __rmul__(self, other):      return Mul.apply(to_tensor(other), self)
    def __rtruediv__(self, other):  return Div.apply(to_tensor(other), self)
    def __rpow__(self, other):      return Pow.apply(to_tensor(other), self)
    def __rmatmul__(self, other):   return Matmul(to_tensor(other), self)

    def __iadd__(self, other):      return Add.apply(self, to_tensor(other))
    def __isub__(self, other):      return self + -other
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


# Tensor Operations
class Neg(Function):
    @staticmethod
    def forward(a):
        return Tensor(-a.data, requires_grad=a.requires_grad)
    
    def backward(ctx, grad_output):
        a = ctx.args
        return -grad_output if a.requires_grad else None

class Add(Function):
    @staticmethod
    def forward(a, b):
        return Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.args 
        return grad if a.requires_grad else None, \
               grad if b.requires_grad else None

class Sub(Function):
    @staticmethod
    def forward(a, b):
        return Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.args
        return grad_output if a.requires_grad else None, \
               -grad_output if b.requires_grad else None

class Mul(Function):
    @staticmethod
    def forward(a, b):
        return Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.args
        grad_a = grad_output * b if a.requires_grad else None 
        grad_b = grad_output * a if b.requires_grad else None 
        return grad_a, grad_b

class Div(Function):
    @staticmethod
    def forward(a, b):
        return Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.args
        grad_a = grad_output / b if a.requires_grad else None 
        grad_b = -grad_output * a / (b**2) if b.requires_grad else None 
        return grad_a, grad_b

class Sum(Function):
    @staticmethod 
    def forward(x, dim, keepdim):
        return Tensor(x.data.sum(axis=dim, keepdims=keepdim))

    @staticmethod 
    def backward(ctx, grad):
        x, _, _ = ctx.args
        return Tensor(np.broadcast_to(grad.data, x.shape)), None

class Max(Function):
    @staticmethod 
    def forward(a, dim, keepdim):
        _data = a.data.max(axis=dim, keepdims=keepdim)
        return Tensor(_data, requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, dim, keepdim = ctx.args
        if not a.requires_grad:
            return None 
        
        mask = (a.data == np.max(a.data, axis=dim, keepdims=keepdim))
        if not keepdim and dim is not None:
            grad_output = np.expand_dims(grad_output, axis=dim)

        grad_a = mask * grad_output
        if not keepdim and dim is not None:
            grad_a = np.sum(grad_a, axis=dim, keepdims=keepdim)
        return grad_a

class Min(Function):
    @staticmethod 
    def forward(a, dim, keepdim):
        _data = a.data.min(axis=dim, keepdims=keepdim)
        return Tensor(_data, requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, dim, keepdim = ctx.args
        if not a.requires_grad:
            return None 
        
        mask = (a.data == np.min(a.data, axis=dim, keepdims=keepdim))
        if not keepdim and dim is not None:
            grad_output = np.expand_dims(grad_output, axis=dim)

        grad_a = mask * grad_output
        if not keepdim and dim is not None:
            grad_a = np.sum(grad_a, axis=dim, keepdims=keepdim)
        return grad_a

class Pow(Function):
    @staticmethod
    def forward(a, b):
        return Tensor(a.data**b.data, requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, b = ctx.args
        if a.requires_grad:
            grad_a = b.data * (a.data ** (b.data - 1)) * grad_output
            return grad_a 
        return None

class Log(Function):
    @staticmethod
    def forward(a):
        return Tensor(np.log(a.data), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a = ctx.args
        if a.requires_grad:
            grad_a = (1. / a.data) * grad_output
            return grad_a 
        return None

class Sqrt(Function):
    @staticmethod
    def forward(a):
        return Tensor(np.sqrt(a.data), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a = ctx.args
        if a.requires_grad:
            grad_a = (1. / 2 * a.data) * grad_output
            return grad_a 
        return None

class Sin(Function):
    @staticmethod
    def forward(a):
        return Tensor(np.sin(a.data), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a = ctx.args
        if a.requires_grad:
            grad_a = np.cos(a.data) * grad_output
            return grad_a 
        return None

class Cos(Function):
    @staticmethod
    def forward(a):
        return Tensor(np.cos(a.data), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a = ctx.args
        if a.requires_grad:
            grad_a = -np.sin(a.data) * grad_output
            return grad_a 
        return None

class Exp(Function):
    @staticmethod
    def forward(a):
        return Tensor(np.exp(a.data), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a = ctx.args
        if a.requires_grad:
            grad_a = np.exp(a.data) * grad_output
            return grad_a 
        return None

class Mean(Function): 
    @staticmethod 
    def forward(a, dim, keepdim):
        return Tensor(a.data.mean(axis=dim, keepdims=keepdim), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, dim, keepdim = ctx.args  

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

class Var(Function):
    @staticmethod 
    def forward(a, dim, keepdim):
        return Tensor(a.data.var(axis=dim, keepdims=keepdim), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, dim, _ = ctx.args 

        if a.requires_grad:
            grad_a = np.ones_like(a.data) * grad_output 
            grad_a = grad_a * 2 * (a.data - a.data.mean(axis=dim, keepdims=True)) / np.prod(np.array(a.shape)[dim])
            return grad_a 
        return None

class Relu(Function):
    @staticmethod 
    def forward(a):
        return Tensor(np.maximum(a.data, 0), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, = ctx.args
        if a.requires_grad:
            grad_a = (a.data > 0) * grad_output 
            return grad_a 
        return None

class Matmul(Function):
    @staticmethod 
    def forward(a, b):
        return Tensor(np.matmul(a.data, b.data), requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, b = ctx.args
        grad_a = grad_output @ b.data.T if a.requires_grad else None 
        grad_b = a.data.T @ grad_output if b.requires_grad else None 
        return grad_a, grad_b

class Transpose(Function):
    @staticmethod 
    def forward(a, *axes):
        return Tensor(np.transpose(a.data, *axes), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, axes = ctx.args
        if a.requires_grad:
            grad_a = np.transpose(grad_output, *axes)
            return grad_a 
        return None

class Slice(Function):
    @staticmethod 
    def forward(a, idx):
        return Tensor(a.data[idx], requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, idx = ctx.args
        if a.requires_grad:
            grad_a = np.zeros_like(a.data)
            grad_a[idx] = grad_output 
            return grad_a 
        return None

class Reshape(Function):
    @staticmethod
    def forward(a, *shape):
        return Tensor(a.data.reshape(shape), requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, shape = ctx.args
        if a.requires_grad:
            grad_a = grad_output.reshape(shape)
            return grad_a 
        return None

class Concat(Function):
    @staticmethod 
    def forward(tensors, dim=0):
        new_data = np.concatenate([t.data for t in tensors], axis=dim)
        requires_grad = any(t.requires_grad for t in tensors)
        return Tensor(new_data, requires_grad=requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        tensors, dim = ctx.args 
        grad_tensors = []
        start = 0
        for tensor in tensors:
            shape = tensor.data.shape
            size = shape[dim]
            grad_tensor = None
            if tensor.requires_grad:
                grad_tensor = grad_output.data.take(indices=np.arange(start, start + size), axis=dim)
                if len(shape) > grad_tensor.ndim:
                    grad_tensor = np.expand_dims(grad_tensor, axis=-1)
            grad_tensors.append(grad_tensor)
            start += size
        return tuple(grad_tensors) 

class Stack(Function):
    @staticmethod
    def forward(tensors, dim=0):
        new_data = np.stack([t.data for t in tensors], axis=dim)
        requires_grad = any(t.requires_grad for t in tensors)
        return Tensor(new_data, requires_grad=requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        tensors, dim = ctx.args 
        grad_tensors = []
        for i, t in enumerate(tensors):
            if t.requires_grad:
                grad_tensor = np.take(grad_output.data, i, axis=dim)
            else:
                grad_tensor = True 
            grad_tensors.append(grad_tensor)
        return tuple(grad_tensors)

class MaskedFill(Function):
    @staticmethod 
    def forward(a, condition, value):
        data = np.where(condition, a.data, value)
        return Tensor(data, requires_grad=a.requires_grad)

    @staticmethod 
    def backward(ctx, grad_output):
        a, condition, _ = ctx.args
        if a.requires_grad:
            grad_a = np.where(condition, grad_output, 0)
            return grad_a 
        return None
