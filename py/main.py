from tensor import *
from layers import *

if __name__=="__main__":
    x = Tensor.rand((1, 10))

    
    print(x)
    print(Tensor.argmax(x))
    print(Tensor.argmin(x))
    print(Tensor.argsort(x))

