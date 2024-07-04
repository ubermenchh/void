from tensor import *
from layers import *

if __name__=="__main__":
    x = Tensor.randn((1, 10), requires_grad=True)
    y = Tensor.randn((1, 10), requires_grad=True)

    z = x.cross_entropy(y)
    z.backward()

    print(x)
    print(y)
    print(z)

    print(x.grad)
    print(y.grad)
