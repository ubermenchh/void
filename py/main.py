from tensor import *
from layers import *

import numpy as np

if __name__=="__main__":
    a = Tensor.randn((3, 4))
    b = Tensor.randn((3, 4))
    c = Tensor.tri(10)
    
    print(a.triu(diag=-1))
    print(b.tril(diag=-1))
    print(c)
