from tensor import *
from layers import *

import numpy as np

if __name__=="__main__":
    a = Tensor.randn((3, 4))
    print(a)
    print(a.max(dim=1))
    print(a.min(dim=1))
