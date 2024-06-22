from tensor import *
from layers import *

import numpy as np

class SimpleNN(Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.l1 = Linear(in_dim, hidden_dim)
        self.relu = ReLU()
        self.l2 = Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

if __name__=="__main__":
    x = Tensor.randn((10, 1), requires_grad=True)
    y = 3 * x + 2 + 0.1 * Tensor.randn((10, 1), requires_grad=True)
    print(y)
    epochs = 10 
    model = SimpleNN(1, 10, 1)
    criterion = MeanSquaredError()
    optimizer = SGD(model.parameters(), lr=0.15)

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        print(f"| Epoch: {epoch} | Loss: {loss.data} |")
    print(y_pred)
