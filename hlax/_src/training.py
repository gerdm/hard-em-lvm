import jax
import optax
from functools import partial
from tqdm.auto import tqdm


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


@partial(jax.vmap, in_axes=(0, 0, 0, 0, None, None, None, None, None, None))
def update_encoder_parameters(key, params_encoder, opt_state, obs, params_decoder, tx,
                              encoder, decoder, grad_loss, num_is_samples=10):
    mll, grads = grad_loss(
        key,
        params_encoder,
        params_decoder,
        obs,
        encoder,
        decoder,
        num_is_samples
    )
    updates, opt_state = tx.update(grads, opt_state, params_encoder)
    params_encoder = optax.apply_updates(params_encoder, updates)
    
    return mll, params_encoder, opt_state


@partial(jax.jit, static_argnames=("tx", "encoder", "decoder", "grad_loss", "num_is_samples"))
def run_epoch_encoder(key, params_encoder, states, observations, tx, params_decoder,
                      encoder, decoder, grad_loss, num_is_samples=10):
    num_obs = len(observations)
    keys_eval = jax.random.split(key, num_obs)
    mll_vals, params_encoder, states = update_encoder_parameters(
        keys_eval, params_encoder, states, observations, params_decoder,
        tx, encoder, decoder, grad_loss, num_is_samples
    )
    
    return mll_vals, params_encoder, states


@partial(jax.vmap, in_axes=(0, None, None))
def init_params_state_encoder(key, encoder, tx):
    key_init, key_sample = jax.random.split(key)
    params = encoder.init(key_init, key_sample)
    state = tx.init(params)
    return params, state


def train_encoder(key, X, encoder, decoder, params_decoder, tx, n_epochs,
                  grad_loss, num_is_samples=10, leave=True):
    """
    Train an unamortised variational distribution q(z|x) using the
    importance-weighted marginal log-likelihood.

    Parameters
    ----------
    grad_loss : function
        jax.value_and_grad of the negative importance-weighted marginal
        log-likelihood.
    """
    n_samples = len(X)
    keys_test, keys_eval = jax.random.split(key)
    keys_test = jax.random.split(keys_test, n_samples)
    keys_eval = jax.random.split(keys_eval, n_epochs)

    states = init_params_state_encoder(keys_test, encoder, tx)
    params_latent, latent_states = states

    hist_negmll = []
    for key_eval in (pbar := tqdm(keys_eval, leave=leave)):
        mll_vals, params_latent, latent_states = run_epoch_encoder(
            key_eval,
            params_latent,
            latent_states,
            X,
            tx,
            params_decoder,
            encoder,
            decoder,
            grad_loss,
            num_is_samples=num_is_samples
        )
        
        mll_mean = mll_vals.mean()
        pbar.set_description(f"{mll_mean=:.3e}")
        hist_negmll.append(mll_mean)

    return {
        "neg_mll": hist_negmll,
        "params": params_latent,
        "states": latent_states
    }
