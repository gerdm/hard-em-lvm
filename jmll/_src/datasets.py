from torchvision.datasets import FashionMNIST

def load_fashion_mnist(melt=True):
    root = "/tmp/fashion-mnist"
    mnist_train = FashionMNIST(root, download=True, train=True)
    mnist_test = FashionMNIST(root, download=True, train=False)
    ...