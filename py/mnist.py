from tensor import *
from layers import * 

import os, gzip, urllib, random 
from tqdm import tqdm

BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4

def fetch_mnist():
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    urls = [
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'
    ]
    
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    for url, file in zip(urls, files):
        file_path = os.path.join(data_folder, file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(url, file_path)

    with gzip.open(os.path.join(data_folder, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)

    with gzip.open(os.path.join(data_folder, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)

    with gzip.open(os.path.join(data_folder, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return x_train, y_train, x_test, y_test


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

def one_hot(labels): return np.eye(10)[labels]
def get_batch(images, labels):
    indices = list(range(0, len(images.data), BATCH_SIZE))
    random.shuffle(indices)
    for i in indices:
        yield images[i:i+BATCH_SIZE], labels[i:i+BATCH_SIZE]

def train(model, train_images, train_labels, optimizer, loss_fn):
    model.train()
    for epoch in range(EPOCHS):
        batch_generator = get_batch(train_images, train_labels)
        num_batches = len(train_images.data) // BATCH_SIZE 
        
        with tqdm(total=num_batches) as pbar:
            for batch_im, batch_lbl in batch_generator:
                optimizer.zero_grad()
                pred = model(batch_im)
                loss = loss_fn(pred, batch_lbl)
                
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix({"Loss": float(loss.data)})

        print(f"Epoch: {epoch}, Loss: {loss.data:.4f}")

if __name__=="__main__":
    x_train, y_train, x_test, y_test = fetch_mnist()
    x_train, y_train = map(Tensor, [x_train, y_train])
    x_test, y_test = map(Tensor, [x_test, y_test])

    model = SimpleNN(28*28, 128, 10)
    optimizer = SGD(model.parameters(), lr=LR)
    loss_fn = CrossEntropyLoss()

    train(model, x_train, y_train, optimizer, loss_fn)
