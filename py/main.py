from tensor import *


if __name__=="__main__":
    a = Tensor([1, 2, 3, 4], requires_grad=True)
    b = Tensor([4, 5, 7, 2], requires_grad=True)
    c = a / b
    d = a * c + (b / a)

    e = d.sum()
    print(e)

    e.backward()
    
    print(e.grad)
    print(d.grad)
    print(c.grad)
    print(b.grad)
    print(a.grad)
