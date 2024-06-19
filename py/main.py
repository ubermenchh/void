from tensor import *
from layers import *

import numpy as np

if __name__=="__main__":
    lin = Linear(3, 4)
    x = randn((4, 3))
    out = lin(x)

    print(out)
