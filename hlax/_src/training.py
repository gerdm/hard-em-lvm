import jax
import distrax
import optax
import jax.numpy as jnp
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


def neg_iwmll_encoder(key, params_encoder, params_decoder, observation,
                  encoder, decoder, num_is_samples=10):
    """
    Importance-weighted marginal log-likelihood for an unamortised, uncoditional
    gaussian encoder.
    """
    latent_samples, (mu_z, std_z) = encoder.apply(params_encoder, key,
                                                  num_samples=num_is_samples)

    _, dim_latent = latent_samples.shape
    # log p(x|z)
    mu_x, logvar_x = decoder.apply(params_decoder, latent_samples)
    std_x = jnp.exp(logvar_x / 2)
    log_px_cond = distrax.MultivariateNormalDiag(mu_x, std_x).log_prob(observation)
    
    # log p(z)
    # mu_z_init, std_z_init = z_prior, jnp.ones(dim_latent)
    mu_z_init, std_z_init = jnp.zeros(dim_latent), jnp.ones(dim_latent)
    log_pz = distrax.MultivariateNormalDiag(mu_z_init, std_z_init).log_prob(latent_samples)
    
    # log q(z)
    log_qz = distrax.MultivariateNormalDiag(mu_z, std_z).log_prob(latent_samples)
    
    # Importance-weighted marginal log-likelihood
    log_prob = log_pz + log_px_cond - log_qz
    niwmll = -jax.nn.logsumexp(log_prob, axis=-1, b=1/num_is_samples)
    
    return niwmll


grad_neg_iwmll_encoder = jax.value_and_grad(neg_iwmll_encoder, argnums=1)


@partial(jax.vmap, in_axes=(0, 0, 0, 0, None, None, None, None, None))
def update_encoder_parameters(key, params_encoder, opt_state, obs, params_decoder, tx,
                      encoder, decoder, num_is_samples=10):
    mll, grads = grad_neg_iwmll_encoder(key, params_encoder, params_decoder, obs,
                                        encoder, decoder, num_is_samples)
    updates, opt_state = tx.update(grads, opt_state, params_encoder)
    params_encoder = optax.apply_updates(params_encoder, updates)
    
    return mll, params_encoder, opt_state


@partial(jax.jit, static_argnames=("tx", "encoder", "decoder", "num_is_samples"))
def run_epoch_encoder(key, params_encoder, states, observations, tx, params_decoder,
                      encoder, decoder, num_is_samples=10):
    num_obs = len(observations)
    keys_eval = jax.random.split(key, num_obs)
    mll_vals, params_encoder, states = update_encoder_parameters(
        keys_eval, params_encoder, states, observations, params_decoder,
        tx, encoder, decoder, num_is_samples
    )
    
    return mll_vals, params_encoder, states