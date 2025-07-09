import numpy as np

def fetch(url):
    import requests,gzip
    dat = requests.get(url).content
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

def load_mnist():
    X_train = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    y_train = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    y_test = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    return X_train, y_train, X_test, y_test

def normalize(X, mean= 0.5, std=0.5):
    # MNIST mean 0.1307 std 0.3081
    X = X/255.
    X = X - mean
    X = X/std

    return X