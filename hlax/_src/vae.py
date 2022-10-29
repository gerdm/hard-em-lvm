import jax
import hlax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames="lossfn")
def train_step(state, X, key, lossfn):
    params = state.params
    apply_fn = state.apply_fn
    
    loss, grads = jax.value_and_grad(lossfn, 1)(key, params, apply_fn, X)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


def train_epoch(key, state, X, batch_size, lossfn):
    num_samples = len(X)
    key_batch, keys_vae = jax.random.split(key)
    batch_ixs = hlax.training.get_batch_train_ixs(key_batch, num_samples, batch_size)
    
    num_batches = len(batch_ixs)
    keys_vae = jax.random.split(keys_vae, num_batches)
    
    total_loss = 0
    for key_vae, batch_ix in zip(keys_vae, batch_ixs):
        X_batch = hlax.training.index_values_batch(X, batch_ix)
        loss, state = train_step(state, X_batch, key_vae, lossfn)
        total_loss += loss
    
    return total_loss.item(), state
