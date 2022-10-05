import jax
import hlax
import optax
import distrax
import jax.numpy as jnp
from tqdm.auto import tqdm
from functools import partial


def initialise_state(key, model, tx_params, tx_latent, X, dim_latent):
    key_init_params, key_init_latent = jax.random.split(key)

    n_train, *_ = X.shape 
    batch_init = jnp.ones((n_train, dim_latent))
    params_decoder = model.init(key_init_params, batch_init)
    z_decoder = jax.random.normal(key_init_latent, (n_train, dim_latent))
    
    opt_params_state = tx_params.init(params_decoder)
    opt_latent_state = tx_latent.init(z_decoder)
    
    target_states = (params_decoder, z_decoder)
    opt_states = (opt_latent_state, opt_params_state)
    
    return opt_states, target_states


@partial(jax.jit, static_argnames=("tx_params", "tx_latent",
                                   "n_its_params", "n_its_latent",
                                   "lossfn", "model"))
def train_step(params_decoder, z_est, opt_states, observations,
               tx_params, tx_latent, n_its_params, n_its_latent,
                lossfn, model):
    opt_latent_state, opt_params_state = opt_states

    grad_e = jax.grad(lossfn, argnums=1)
    grad_m = jax.grad(lossfn, argnums=0)
    
    # E-step
    for i in range(n_its_latent):
        grad_z = grad_e(params_decoder, z_est, observations, model)
        updates, opt_latent_state = tx_latent.update(grad_z, opt_latent_state, z_est)
        z_est = optax.apply_updates(z_est, updates)
    
    # M-step
    for i in range(n_its_params):
        grad_theta = grad_m(params_decoder, z_est, observations, model)
        updates, opt_params_state = tx_params.update(grad_theta, opt_params_state, params_decoder)
        params_decoder = optax.apply_updates(params_decoder, updates)

    nll = lossfn(params_decoder, z_est, observations, model) 
    new_states = (opt_latent_state, opt_params_state)
    return nll, params_decoder, z_est, new_states


def train_epoch_full(key, observations, model, tx, dim_latent, lossfn, n_its, n_epochs):
    """
    Train a full-batch hard-EM latent variable model of the form
    p(x, z) = p(x|z) p(z) = N(x|f(z), sigma^2 I) N(z|0, I),
    where sigma^2  is a vector of diagonal elements of the covariance matrix.

    In this implementation, we consider a single optimiser for both the
    parameters and the latent variables. We also consider the same
    number of iterations for both the E-step and the M-step.

    Parameters
    ----------
    key: jax.random.PRNGKey
        Random number generator key
    observations: jnp.ndarray
        Data matrix of shape (n_samples, n_features)
    model: flax.nn.Module
        Decoder model (input z, output x)
    tx: optax.GradientTransformation
        Optimiser
    dim_latent: int
        Dimensionality of latent space
    lossfn: function
        Loss function taking the form
        lossfn(params, z_batch, X_batch, model)
        where we seek to approximate
        X_batch = model.apply(params, z_batch)
    n_its: int
        Number of iterations of the E-step and M-step
    n_epochs: int
        Number of epochs to train for
    """
    opt_states, target_states = initialise_state(key, model, tx, tx,
                                                 observations, dim_latent)
    params_decoder, z_decoder = target_states

    nll_hist = []
    for e in tqdm(range(n_epochs)):
        res = train_step(params_decoder, z_decoder, opt_states,
                        tx_params=tx, tx_latent=tx,
                        n_its_params=n_its, n_its_latent=n_its,
                        lossfn=lossfn, model=model, observations=observations)
        nll, params_decoder, z_decoder, opt_states = res
        nll_hist.append(nll.item())
        print(f"{nll:0.4e}", end="\r")
    return nll_hist, params_decoder, z_decoder


def train_epoch(key, params, z_est, opt_states, observations,
                batch_size, model, tx_params, tx_latent,
                n_its_params, n_its_latent, lossfn):
    """
    Hard-EM LVM mini-batch training epoch

    Parameters
    ----------
    n_its_params: int
        Number of iterations of the M-step
    n_its_latent: int
        Number of iterations of the E-step
    """
    num_samples = len(observations)
    key_batch, keys_vae = jax.random.split(key)
    batch_ixs = hlax.training.get_batch_train_ixs(key_batch, num_samples, batch_size)
    num_batches = len(batch_ixs)
    keys_vae = jax.random.split(keys_vae, num_batches)
    total_nll = 0
    for batch_ix in batch_ixs:
        batch = observations[batch_ix, ...]
        z_batch = z_est[batch_ix, ...]

        res = train_step(params, z_batch, opt_states,
                        tx_params=tx_params, tx_latent=tx_latent,
                        n_its_params=n_its_params, n_its_latent=n_its_latent,
                        lossfn=lossfn, model=model, observations=batch)
        nll, params, z_batch, opt_states = res
        # Update minibatch of latent variables
        z_est = z_est.at[batch_ix, ...].set(z_batch)

        total_nll += nll
    return total_nll, params, z_est, opt_states


def loss_hard_nmll(params, z_batch, X_batch, model):
    """
    Loss function
    -------------
    
    Negative Marginal log-likelihood for hard EM
    assuming an isotropic Gaussian prior with zero mean
    and a decoder with a diagonal covariance matrix

    Parameters
    ----------
    params: pytree
        Parameters of the decoder model, i.e.,
        model.apply(params, z_batch) = X_batch (approx)
    z_batch: jnp.ndarray
        Batch of latent variables
    X_batch: jnp.ndarray
        Batch of observations
    model: flax.nn.Module
        Decoder model (input z, output x)
    """
    dim_latent = model.dim_latent
    
    mean_x, logvar_x = model.apply(params, z_batch)
    std_x = jnp.exp(logvar_x / 2)
    
    dist_prior = distrax.MultivariateNormalDiag(jnp.zeros(dim_latent), jnp.ones(dim_latent))
    dist_decoder = distrax.MultivariateNormalDiag(mean_x, std_x)
    
    log_prob_z_prior = dist_prior.log_prob(z_batch)
    log_prob_x = dist_decoder.log_prob(X_batch)
    
    log_prob = log_prob_z_prior + log_prob_x
    
    return -log_prob.sum()
