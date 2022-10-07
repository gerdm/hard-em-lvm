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


def e_step(_, state, grad_e, observations, tx, model):
    """
    E-step of the hard-EM LVM
    """
    opt_state, params, z_est = state
    grad_z = grad_e(params, z_est, observations, model)
    updates, opt_state = tx.update(grad_z, opt_state, z_est)
    z_est = optax.apply_updates(z_est, updates)
    new_state = (opt_state, params, z_est)
    return new_state


def m_step(_, state, grad_m, observations, tx, model):
    """
    M-step of the hard-EM LVM
    """
    opt_state, params, z_est = state
    grad_params = grad_m(params, z_est, observations, model)
    updates, opt_state = tx.update(grad_params, opt_state, params)
    params = optax.apply_updates(params, updates)
    new_state = (opt_state, params, z_est)
    return new_state


@partial(jax.jit, static_argnames=("tx_params", "tx_latent",
                                   "n_its_params", "n_its_latent",
                                   "lossfn", "model"))
def train_step(params_decoder, z_est, opt_states, observations,
               tx_params, tx_latent, n_its_params, n_its_latent,
                lossfn, model):
    opt_latent_state, opt_params_state = opt_states

    grad_e = jax.grad(lossfn, argnums=1)
    grad_m = jax.grad(lossfn, argnums=0)

    part_e_step = partial(e_step,
                          grad_e=grad_e, observations=observations,
                          tx=tx_latent, model=model)
    part_m_step = partial(m_step,
                            grad_m=grad_m, observations=observations,
                            tx=tx_params, model=model)

    # E-step
    init_state = (opt_latent_state, params_decoder, z_est)
    final_state = jax.lax.fori_loop(0, n_its_latent, part_e_step, init_state)
    opt_latent_state, params_decoder, z_est = final_state

    # M-step
    init_state = (opt_params_state, params_decoder, z_est)
    final_state = jax.lax.fori_loop(0, n_its_params, part_m_step, init_state)
    opt_params_state, params_decoder, z_est = final_state

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


@jax.jit
def update_latent_params(z_total, z_sub, ix_sub):
    """
    Update the latent variables and parameters of a batch
    """
    z_total = z_total.at[ix_sub].set(z_sub)
    return z_total


@jax.jit
def get_batch_adam_params(opt_latent, ixs):
    """
    Get the parameters of a batch from the optimiser state
    """
    mu_sub = jax.tree_map(lambda x: x[ixs], opt_latent[0].mu)
    nu_sub = jax.tree_map(lambda x: x[ixs], opt_latent[0].nu)

    return mu_sub, nu_sub


@jax.jit
def create_batch_adam_params(opt_state, ixs):
    """
    Create the optimiser state from the parameters of a batch
    """
    opt_latent_state, opt_params_state = opt_state
    mu_sub, nu_sub = get_batch_adam_params(opt_latent_state, ixs)

    opt_latent_sub = (
        opt_latent_state[0]._replace(
            mu = mu_sub,
            nu = nu_sub
        ),
    ) + opt_latent_state[1:]

    opt_state_sub = (opt_latent_sub, opt_params_state)
    return opt_state_sub


@jax.jit
def reconstruct_full_adam_params(opt_state, opt_state_sub, ixs):
    opt_latent_state, _ = opt_state
    opt_latent_batch, opt_params_update = opt_state_sub
    mu_sub, nu_sub = get_batch_adam_params(opt_latent_batch, ixs)

    mu_update = jax.tree_map(lambda x: x.at[ixs].set(mu_sub), opt_latent_state[0].mu)
    nu_update = jax.tree_map(lambda x: x.at[ixs].set(nu_sub), opt_latent_state[0].nu)

    opt_latent_state_new = (
        opt_latent_state[0]._replace(
            mu = mu_update,
            nu = nu_update
        ),
    ) + opt_latent_state[1:]

    opt_state_new = (opt_latent_state_new, opt_params_update)
    return opt_state_new


def train_epoch_adam(key, params, z_est, opt_states, observations,
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
        batch, z_batch = hlax.training.index_values_latent_batch(observations, z_est, batch_ix)

        # Decompose batch params for E-step
        opt_states_batch = create_batch_adam_params(opt_states, batch_ix)

        res = train_step(params, z_batch, opt_states_batch,
                        tx_params=tx_params, tx_latent=tx_latent,
                        n_its_params=n_its_params, n_its_latent=n_its_latent,
                        lossfn=lossfn, model=model, observations=batch)
        nll, params, z_batch, opt_states_batch = res
        # Update minibatch of latent variables
        z_est = update_latent_params(z_est, z_batch, batch_ix)

        # Update minibatch of optimiser states
        opt_states = reconstruct_full_adam_params(opt_states, opt_states_batch, batch_ix)

        total_nll += nll
    return total_nll, params, z_est, opt_states


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
        batch, z_batch = hlax.training.index_values_latent_batch(observations, z_est, batch_ix)

        res = train_step(params, z_batch, opt_states,
                        tx_params=tx_params, tx_latent=tx_latent,
                        n_its_params=n_its_params, n_its_latent=n_its_latent,
                        lossfn=lossfn, model=model, observations=batch)
        nll, params, z_batch, opt_states = res
        # Update minibatch of latent variables
        z_est = update_latent_params(z_est, z_batch, batch_ix)

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
    
    return -log_prob.mean()
