import jax
from functools import partial


@jax.jit
def index_values_latent_batch(observations, latent, ixs):
    """
    Index values of a batch of observations and latent variables
    """
    X_batch = observations[ixs]
    z_batch = latent[ixs]
    return X_batch, z_batch


@jax.jit
def index_values_batch(observations, ixs):
    """
    Index values of a batch of observations
    """
    X_batch = observations[ixs]
    return X_batch


@partial(jax.jit, static_argnums=(1,2))
def get_batch_train_ixs(key, num_samples, batch_size):
    """
    Obtain the training indices to be used in an epoch of
    mini-batch optimisation.
    """
    steps_per_epoch = num_samples // batch_size
    
    batch_ixs = jax.random.permutation(key, num_samples)
    batch_ixs = batch_ixs[:steps_per_epoch * batch_size]
    batch_ixs = batch_ixs.reshape(steps_per_epoch, batch_size)
    
    return batch_ixs
