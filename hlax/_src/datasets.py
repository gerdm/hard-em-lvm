import jax
import jax.numpy as jnp
from torchvision.datasets import FashionMNIST, MNIST, Omniglot

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


def load_mnist(n_train, n_test, melt=True, normalize=True, root=None):
    root = "/tmp/mnist" if root is None else root
    mnist_train = MNIST(root, download=True, train=True)
    mnist_test = MNIST(root, download=True, train=False)

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


def load_omniglot(key, n_train, n_test, melt=True, normalize=True, root=None):
    root = "/tmp/omniglot" if root is None else root
    key_train, key_test = jax.random.split(key)
    dataset_train = Omniglot(root, background=True, download=True, transform=jnp.array)
    dataset_test = Omniglot(root, background=False, download=True, transform=jnp.array)

    ixs_train = jax.random.choice(key_train, n_train, (n_train,), replace=False).to_py()    
    ixs_test = jax.random.choice(key_test, n_test, (n_test,), replace=False).to_py()    

    X_train = jnp.array([dataset_train[ix][0] for ix in ixs_train])
    X_test = jnp.array([dataset_test[ix][0] for ix in ixs_test])

    _, nc, nc = X_train.shape
    if melt:
        X_train = X_train.reshape(-1, nc ** 2)
        X_test = X_test.reshape(-1, nc ** 2)
    
    xmax = X_train.max()
    X_train = X_train / xmax
    X_test = X_test / xmax

    if normalize:
        xmean = X_train.mean()
        xstd = X_train.std()

        X_train = (X_train - xmean) / xstd
        X_test = (X_test - xmean) / xstd

    return (X_train, None), (X_test, None)