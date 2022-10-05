import jax
import hlax
import distrax
import jax.numpy as jnp
from functools import partial

def iwae(key, params, apply_fn, X_batch):
    """
    Loss function
    Importance-weighted Variational Autoencoder (IW-VAE)
    """
    batch_size = len(X_batch)
    keys = jax.random.split(key, batch_size)
    
    encode_decode = jax.vmap(apply_fn, (None, 0, 0))
    encode_decode = encode_decode(params, X_batch, keys)
    z, (mean_z, logvar_z), (mean_x, logvar_x) = encode_decode
    _, num_is_samples, dim_latent = z.shape
    
    std_z = jnp.exp(logvar_z / 2)
    std_x = jnp.exp(logvar_x / 2)
    
    dist_prior = distrax.MultivariateNormalDiag(jnp.zeros(dim_latent),
                                                jnp.ones(dim_latent))
    dist_decoder = distrax.MultivariateNormalDiag(mean_x, std_x)
    dist_posterior = distrax.Normal(mean_z[:, None, :], std_z[:, None, :])
    
    log_prob_z_prior = dist_prior.log_prob(z)
    log_prob_x = dist_decoder.log_prob(X_batch[:, None, :])
    log_prob_z_post = dist_posterior.log_prob(z).sum(axis=-1)
    
    log_prob = log_prob_z_prior + log_prob_x - log_prob_z_post
    
    # negative Importance-weighted marginal log-likelihood
    niwmll = -jax.nn.logsumexp(log_prob, axis=-1, b=1/num_is_samples).sum()
    return niwmll

@partial(jax.jit, static_argnames="lossfn")
def train_step(state, X, key, lossfn):
    params = state.params
    apply_fn = state.apply_fn
    
    loss, grads = jax.value_and_grad(lossfn, 1)(key, params, apply_fn, X)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


def train_epoch(key, state, X, batch_size):
    num_samples = len(X)
    key_batch, keys_vae = jax.random.split(key)
    batch_ixs = hlax.training.get_batch_train_ixs(key_batch, num_samples, batch_size)
    
    num_batches = len(batch_ixs)
    keys_vae = jax.random.split(keys_vae, num_batches)
    
    total_loss = 0
    for key_vae, batch_ix in zip(keys_vae, batch_ixs):
        X_batch = X[batch_ix, ...]
        loss, state = train_step(state, X_batch, key_vae, lossfn=iwae)
        total_loss += loss
    
    return total_loss.item(), state