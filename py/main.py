from tensor import *
from layers import *


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

    data_x = Tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], requires_grad=True)
    data_y = Tensor([
        [1], 
        [1], 
        [1], 
        [0]
    ], requires_grad=True)
    

    epochs = 100 
    model = SimpleNN(2, 10, 1)
    criterion = MeanSquaredError()
    optimizer = SGD(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(data_x)
        loss = criterion(y_pred, data_y)
        loss.backward()
        optimizer.step()

        print(f"| Epoch: {epoch} | Loss: {loss.data} |")

    print(y_pred)
