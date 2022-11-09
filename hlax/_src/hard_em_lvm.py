"""
Hard-EM Latent Variable Model training
"""
import jax
import hlax
import chex
import optax
import flax.linen as nn
import jax.numpy as jnp
from time import time
from tqdm.auto import tqdm
from functools import partial
from typing import Callable, Dict
from dataclasses import dataclass

@dataclass
class CheckpointsConfig:
    model_decoder: nn.Module

    num_epochs: int
    batch_size: int
    dim_latent: int
    eval_epochs: list

    num_its_params: int
    num_its_latent: int

    tx_params: optax.GradientTransformation
    tx_latent: optax.GradientTransformation


def load_config(
    dict_config: Dict,
    model: nn.Module,
) -> CheckpointsConfig:
    """
    Load a CheckpointsConfig object from a dictionary
    """
    learning_rate = dict_config["train"]["learning_rate"]

    tx_params = optax.adam(learning_rate)
    tx_latent = optax.adam(learning_rate)

    config = hlax.hard_em_lvm.CheckpointsConfig(
        num_epochs=dict_config["train"]["num_epochs"],
        batch_size=dict_config["train"]["batch_size"],
        dim_latent=dict_config["setup"]["dim_latent"],
        eval_epochs=dict_config["train"]["eval_epochs"],
        num_its_params=dict_config["train"]["hard_em"]["num_its_params"],
        num_its_latent=dict_config["train"]["hard_em"]["num_its_latent"],
        tx_params=tx_params,
        tx_latent=tx_latent,
        model_decoder=model,
    )
    return config


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

    # E-step (n_its_latent iterations)
    init_state = (opt_latent_state, params_decoder, z_est)
    final_state = jax.lax.fori_loop(0, n_its_latent, part_e_step, init_state)
    opt_latent_state, params_decoder, z_est = final_state

    # M-step (n_its_params iterations)
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
            mu=mu_update,
            nu=nu_update
        ),
    ) + opt_latent_state[1:]

    opt_state_new = (opt_latent_state_new, opt_params_update)
    return opt_state_new


def train_epoch_adam(key, params, z_est, opt_states, observations,
                batch_size, model, tx_params, tx_latent,
                n_its_params, n_its_latent, lossfn):
    """
    Hard-EM LVM mini-batch training epoch

    See https://flax.readthedocs.io/en/latest/guides/model_surgery.html#surgery-with-optimizers
    for details on how to update the optimiser state when the parameters to be optimised
    depend on the batch.

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
    return total_nll / num_batches, params, z_est, opt_states


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


def train_checkpoints(
    key: chex.ArrayDevice,
    config: CheckpointsConfig,
    X: chex.ArrayDevice,
    lossfn: Callable,
):
    """
    Find inference model parameters theta
    using the Hard EM algorithm
    """
    dict_params = {}
    dict_times = {}
    hist_loss = []
    decoder = config.model_decoder

    key_init, key_step = jax.random.split(key)
    keys_step = jax.random.split(key_step, config.num_epochs)

    states = initialise_state(
        key_init,
        decoder,
        config.tx_params,
        config.tx_latent,
        X,
        config.dim_latent,
    )
    opt_states, target_states = states
    params_decoder, z_est = target_states

    time_init = time()
    pbar = tqdm(enumerate(keys_step), total=config.num_epochs)
    for e, keyt in pbar:
        res = train_epoch_adam(
            keyt,
            params_decoder,
            z_est,
            opt_states,
            X,
            config.batch_size,
            decoder,
            config.tx_params, config.tx_latent,
            config.num_its_params, config.num_its_latent,
            lossfn
        )
        loss, params_decoder, z_est, opt_states = res
        hist_loss.append(loss)
        pbar.set_description(f"hEM-{loss=:.3e}")

        if (enum := e + 1) in config.eval_epochs:
            time_ellapsed = time() - time_init
            dict_times[f"e{enum}"] = time_ellapsed
            dict_params[f"e{enum}"] = params_decoder

    output = {
        "times": dict_times,
        "checkpoint_params": dict_params,
        "hist_loss": jnp.array(hist_loss),
    }
    return output
