import jax.numpy as jnp
from torchvision.datasets import FashionMNIST

def load_fashion_mnist(n_train, n_test, melt=True, normalize=True, root=None):
    root = "/tmp/fashion-mnist" if root is None else root
    mnist_train = FashionMNIST(root, download=True, train=True)
    mnist_test = FashionMNIST(root, download=True, train=False)

    y_train = jnp.array(mnist_train.targets)[:n_train]
    X_train = jnp.array(mnist_train.data)[:n_train]
    if melt:
        X_train = X_train.reshape(-1, 28 ** 2)

    y_test = jnp.array(mnist_test.targets)[:n_test]
    X_test = jnp.array(mnist_test.data)[:n_test]

    if melt:
        X_test = X_test.reshape(-1, 28 ** 2)

    xmax = X_train.max()
    X_train = X_train / xmax
    X_test = X_test / xmax

    if normalize:
        xmean = X_train.mean()
        xstd = X_train.std()

        X_train = (X_train - xmean) / xstd
        X_test = (X_test - xmean) / xstd

    return (X_train, y_train), (X_test, y_test)
