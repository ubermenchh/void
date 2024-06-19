from tensor import *


if __name__=="__main__":
    a = randn_tensor((3, 4), True)
    b = randn_tensor((3, 4), True)
    c = a @ b.T
    
    print(c)

    c.backward()
    print(a.grad)
    print(b.grad)
