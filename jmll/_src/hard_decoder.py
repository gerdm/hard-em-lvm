import jax
import optax
import distrax
import jax.numpy as jnp
from tqdm.auto import tqdm
from functools import partial


def initialise_epoch(key, model, tx, X, dim_latent):
    key_init_params, key_init_latent = jax.random.split(key)

    n_train, *_ = X.shape 
    batch_init = jnp.ones((n_train, dim_latent))
    params_decoder = model.init(key_init_params, batch_init)
    z_decoder = jax.random.normal(key_init_latent, (n_train, dim_latent))
    
    opt_params_state = tx.init(params_decoder)
    opt_latent_state = tx.init(z_decoder)
    
    target_states = (params_decoder, z_decoder)
    opt_states = (opt_latent_state, opt_params_state)
    
    return opt_states, target_states


def epoch_step(params_decoder, z_est, opt_states, observations, tx, lossfn, model, n_its=1):
    opt_latent_state, opt_params_state = opt_states

    grad_e = jax.grad(lossfn, argnums=1)
    grad_m = jax.grad(lossfn, argnums=0)
    
    # E-step
    for i in range(n_its):
        grad_z = grad_e(params_decoder, z_est, observations, model)
        updates, opt_latent_state = tx.update(grad_z, opt_latent_state, z_est)
        z_est = optax.apply_updates(z_est, updates)
    
    # M-step
    for i in range(n_its):
        nll, grad_theta = grad_m(params_decoder, z_est, observations, model)
        updates, opt_params_state = tx.update(grad_theta, opt_params_state, params_decoder)
        params_decoder = optax.apply_updates(params_decoder, updates)

    nll = lossfn(params_decoder, z_est, observations, model) 
    new_states = (opt_latent_state, opt_params_state)
    return nll, params_decoder, z_est, new_states


def train_epoch_full(key, X, model, tx, dim_latent, lossfn, n_its, n_epochs):
    """
    Train a full-batch hard-EM latent variable model of the form
    p(x, z) = p(x|z) p(z) = N(x|f(z), sigma^2 I) N(z|0, I),
    where sigma^2  is a vector of diagonal elements of the covariance matrix.

    Parameters
    ----------
    key: jax.random.PRNGKey
        Random number generator key
    X: jnp.ndarray
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
    opt_states, target_states = initialise_epoch(key, model, tx, X, dim_latent)
    params_decoder, z_decoder = target_states

    part_step = jax.jit(partial(epoch_step,
                                tx=tx, n_its=n_its, 
                                lossfn=lossfn,
                                model=model))
    nll_hist = []
    for e in tqdm(range(n_epochs)):
        nll, params_decoder, z_decoder, opt_states = part_step(params_decoder, z_decoder, opt_states)
        nll_hist.append(nll.item())
        print(f"{nll:0.4e}", end="\r")
    return nll_hist, params_decoder, z_decoder


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
    batch_size = len(X_batch)
    dim_latent = model.dim_latent
    
    mean_x, logvar_x = model.apply(params, z_batch)
    std_x = jnp.exp(logvar_x / 2)
    
    dist_prior = distrax.MultivariateNormalDiag(jnp.zeros(dim_latent), jnp.ones(dim_latent))
    dist_decoder = distrax.MultivariateNormalDiag(mean_x, std_x)
    
    log_prob_z_prior = dist_prior.log_prob(z_batch)
    log_prob_x = dist_decoder.log_prob(X_batch)
    
    log_prob = log_prob_z_prior + log_prob_x
    
    return -log_prob.sum()