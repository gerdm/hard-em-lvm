"""
Unamortised VAE
"""

import jax
import chex 
import hlax
import optax
import jax.numpy as jnp
import flax.linen as nn
from time import time
from tqdm.auto import tqdm
from functools import partial
from typing import Callable
from flax.core import freeze, unfreeze
from flax.training.train_state import TrainState


class CheckpointsConfig:
    model: nn.Module
    num_epochs: int
    batch_size: int
    dim_latent: int
    eval_epochs: list
    tx: optax.GradientTransformation
    num_is_samples: int


@jax.jit
def get_batch_adam_params_encoder(opt_state, ixs):
    """
    Get mu and nu optimiser parameters
    """
    encoder_sub_mu = opt_state[0].mu["params"]["encoder"]
    encoder_sub_nu = opt_state[0].nu["params"]["encoder"]
    
    encoder_sub_mu = jax.tree_map(lambda x: x[ixs], encoder_sub_mu)
    encoder_sub_nu = jax.tree_map(lambda x: x[ixs], encoder_sub_nu)
    
    return encoder_sub_mu, encoder_sub_nu


@jax.jit
def create_batch_adam_params(opt_state, ixs):
    mu_sub, nu_sub = get_batch_adam_params_encoder(opt_state, ixs)
    
    opt_state_batch = opt_state[0]
    mu_params = unfreeze(opt_state_batch.mu)
    nu_params = unfreeze(opt_state_batch.nu)

    # Replace encoder opt params with indexed params
    mu_params["params"]["encoder"] = mu_sub
    nu_params["params"]["encoder"] = nu_sub
    
    opt_state_batch = opt_state_batch._replace(
        mu=freeze(mu_params),
        nu=freeze(nu_params)
    )
    
    opt_state_batch = (
        opt_state_batch,
    ) + opt_state[1:]
    
    return opt_state_batch


@jax.jit
def update_pytree(pytree, pytree_subset, ixs):
    """
    Update the subset of a pytree.
    
    Parameters
    ----------
    pytree:
        Target pytree
    pytree_subset
        Values of the pytree to update
    ixs: DeviceArray
        Indices mapping from the subset to the
        full pytree
    """
    pytree_update = jax.tree_map(
        lambda vfull, vsub: vfull.at[ixs].set(vsub),
        pytree, pytree_subset
    )
    return pytree_update


@jax.jit
def create_state_batch(state, ixs):
    """
    Create a batch of the unamortised TrainStep
    """
    params_batch_encoder = jax.tree_map(lambda x: x[ixs], state.params["params"]["encoder"])
    params_batch = freeze({
        "params": {
            "encoder": params_batch_encoder,
            "decoder": state.params["params"]["decoder"]
        }
    })
    
    opt_state_batch = create_batch_adam_params(state.opt_state, ixs)
    
    state_batch = TrainState(
        step=state.step,
        apply_fn=state.apply_fn,
        tx=state.tx,
        params=params_batch,
        opt_state=opt_state_batch,
    )
    
    return state_batch


@partial(jax.jit, static_argnames=("lossfn",))
def update_state_batch(key, X_batch, state_batch, lossfn):
    loss_valgrad = jax.value_and_grad(lossfn, 1)
    loss, grads_batch = loss_valgrad(key, state_batch.params, state_batch.apply_fn, X_batch)
    new_state_batch = state_batch.apply_gradients(grads=grads_batch)
    return loss, new_state_batch


@jax.jit
def reconstruct_params(state, new_state_batch, ixs):
    params_encoder_update = unfreeze(state.params["params"]["encoder"])
    params_batch_encoder_update = unfreeze(new_state_batch.params["params"]["encoder"])
    params_encoder_update = update_pytree(params_encoder_update, params_batch_encoder_update, ixs)

    params_decoder_update = unfreeze(new_state_batch.params["params"]["decoder"])

    params_update = freeze({
        "params": {
            "encoder": params_encoder_update,
            "decoder": params_decoder_update,
        }
    })

    return params_update


@jax.jit
def reconstruct_opt_state(state, new_state_batch, ixs):
    mu_state = unfreeze(state.opt_state[0].mu)
    nu_state = unfreeze(state.opt_state[0].nu)


    # Update decoder optimisation-state params
    mu_state["params"]["decoder"] = new_state_batch.opt_state[0].mu["params"]["decoder"]
    nu_state["params"]["decoder"] = new_state_batch.opt_state[0].nu["params"]["decoder"]

    # Update encoder optimisation-state params
    batch_mu_encoder = unfreeze(new_state_batch.opt_state[0].mu["params"]["encoder"])
    mu_state["params"]["encoder"] = update_pytree(mu_state["params"]["encoder"], batch_mu_encoder, ixs)

    batch_nu_encoder = unfreeze(new_state_batch.opt_state[0].nu["params"]["encoder"])
    nu_state["params"]["encoder"] = update_pytree(nu_state["params"]["encoder"], batch_nu_encoder, ixs)

    mu_state = freeze(mu_state)
    nu_state = freeze(nu_state)

    opt_state_update = (
        state.opt_state[0]._replace(
            mu=mu_state,
            nu=nu_state,
        ),
    ) + state.opt_state[1:]

    return opt_state_update


@jax.jit
def update_reconstruct_state(state, new_params, new_opt_state):
    new_state = TrainState(
        step=state.step,
        apply_fn=state.apply_fn,
        tx=state.tx,
        params=new_params,
        opt_state=new_opt_state,
    )
    return new_state


@partial(jax.jit, static_argnames=("lossfn",))
def train_step_ix(key, X, state, ixs, lossfn):
    X_batch = X[ixs]
    state_batch = create_state_batch(state, ixs)
    loss, new_state_batch = update_state_batch(key, X_batch, state_batch, lossfn)
    new_params = reconstruct_params(state, new_state_batch, ixs)
    new_opt_state = reconstruct_opt_state(state, new_state_batch, ixs)
    new_state = update_reconstruct_state(state, new_params, new_opt_state)
    return loss, new_state


def train_epoch(key, X, state, batch_size, lossfn):
    num_obs = X.shape[0]
    key_batch, key_train = jax.random.split(key)
    batch_ixs = hlax.training.get_batch_train_ixs(key_batch, num_obs, batch_size)
    num_batches = len(batch_ixs)
    keys_train = jax.random.split(key_train, num_batches)

    losses = 0
    for batch_ix, key_epoch in zip(batch_ixs, keys_train):
        loss, state = train_step_ix(key_epoch, X, state, batch_ix, lossfn)
        losses += loss

    return losses / num_batches, state


def train_checkpoints(
    key: chex.ArrayDevice,
    config: CheckpointsConfig,
    X: chex.ArrayDevice,
    lossfn: Callable,
):
    """
    Find inference model parameters theta at multiple
    epochs.
    """
    dict_params = {}
    dict_times = {}
    hist_loss = []
    _, *dim_obs = X.shape

    key_params_init, key_eps_init, key_train = jax.random.split(key, 3)
    keys_train = jax.random.split(key_train, config.num_epochs)

    batch_init = jnp.ones((config.batch_size, *dim_obs))
    params_init = config.model.init(key_params_init, batch_init, key_eps_init, num_samples=3)

    state = TrainState(
        apply_fn=partial(config.model.apply, num_samples=config.num_is_samples),
        params=params_init,
        tx=config.tx,
    )

    time_init = time()
    for e, keyt in (pbar := tqdm(enumerate(keys_train), total=config.num_epochs)):
        loss, state = hlax.unamortised.train_epoch(keyt, X, state, config.batch_size, lossfn)

        hist_loss.append(loss)
        pbar.set_description(f"loss={loss:0.5e}")

        if (enum := e + 1) in config.eval_epochs:
            time_ellapsed = time() - time_init
            params_decoder = freeze({"params": unfreeze(state.params)["params"]["decoder"]})

            dict_times[f"e{enum}"] = time_ellapsed
            dict_params[f"e{enum}"] = params_decoder
    
    output = {
        "times": dict_times,
        "checkpoint_params": dict_params,
        "hist_loss": jnp.array(hist_loss),
    }
    return output
