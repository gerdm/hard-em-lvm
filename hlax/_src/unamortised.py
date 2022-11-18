"""
Unamortised VAE
"""

import jax
import chex
import hlax
import optax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
from time import time
from tqdm.auto import tqdm
from functools import partial
from typing import Callable, Dict
from flax.core import freeze, unfreeze
from flax.training.train_state import TrainState


@dataclass
class Config:
    num_epochs: int
    batch_size: int
    num_e_steps: int


@dataclass
class CheckpointsConfig(Config):
    eval_epochs: list

def load_config(
    dict_config: Dict,
    model: nn.Module,
) -> CheckpointsConfig:
    """
    Load config from dictionary.
    """
    learning_rate = dict_config["train"]["learning_rate"]
    tx = optax.adam(learning_rate)

    config = CheckpointsConfig(
        num_epochs=dict_config["train"]["num_epochs"],
        batch_size=dict_config["train"]["batch_size"],
        dim_latent=dict_config["setup"]["dim_latent"],
        eval_epochs=dict_config["train"]["eval_epochs"],
        num_is_samples=dict_config["train"]["vae"]["num_is_samples"],
        num_m_steps=dict_config["train"]["hard_em"]["num_its_params"],
        num_e_steps=dict_config["train"]["hard_em"]["num_its_latent"],
        tx=tx,
        model=model,
    )
    return config


@jax.jit
def create_batch_adam_params(opt_state, ixs):
    sub_mu = opt_state[0].mu
    sub_nu = opt_state[0].nu

    sub_mu = jax.tree_map(lambda x: x[ixs], sub_mu)
    sub_nu = jax.tree_map(lambda x: x[ixs], sub_nu)

    opt_state_batch = opt_state[0]
    opt_state_batch = opt_state_batch._replace(
        mu=sub_mu,
        nu=sub_nu
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
def create_state_encoder_batch(state_encoder, ixs):
    """
    Create a TrainState object for a batch of observations.
    We create this train state with
    1. The subset of parameters for the batch
    2. The subset of optimiser parameters for the batch
    """
    params_batch_encoder = jax.tree_map(lambda x: x[ixs], state_encoder.params)
    opt_state_batch = create_batch_adam_params(state_encoder.opt_state, ixs)

    state_batch = TrainState(
        step=state_encoder.step,
        apply_fn=state_encoder.apply_fn,
        tx=state_encoder.tx,
        params=params_batch_encoder,
        opt_state=opt_state_batch,
    )

    return state_batch


@partial(jax.jit, static_argnames=("lossfn",))
def update_state_batch(key, X_batch, state_batch, lossfn):
    loss_valgrad = jax.value_and_grad(lossfn, 1)
    loss, grads_batch = loss_valgrad(key, state_batch.params, state_batch.apply_fn, X_batch)
    new_state_batch = state_batch.apply_gradients(grads=grads_batch)
    return loss, new_state_batch


@partial(jax.jit, static_argnames=("entry_name",))
def adam_replace_opt_state(state, old_mu, old_nu, entry_name):
    opt_state_update = state.opt_state
    mu_update = unfreeze(opt_state_update[0].mu)
    nu_update = unfreeze(opt_state_update[0].nu)

    mu_update["params"][entry_name] = old_mu
    nu_update["params"][entry_name] = old_nu

    opt_state_update = (
        opt_state_update[0]._replace(
            mu=freeze(mu_update),
            nu=freeze(nu_update)
        ),
    ) + opt_state_update[1:]

    state_update = state.replace(opt_state=opt_state_update)
    return state_update


@partial(jax.jit, static_argnames=("lossfn",))
def e_step(
    _: int,
    state_encoder: TrainState,
    state_decoder: TrainState,
    lossfn: Callable,
    X: chex.ArrayDevice
):
    params_encoder = state_encoder.params
    params_decoder = state_decoder.params
    grads_encoder = jax.grad(lossfn, 0)(params_encoder, params_decoder, X)

    state_encoder = state_encoder.apply_gradients(grads=grads_encoder)
    return state_encoder


@jax.jit
def m_step(state_decoder, grads_decoder, num_batches):
    """
    Update the decoder parameters after accumulating
    gradients over a number of batches in the E-step.
    """
    grads_decoder = jax.tree_map(lambda x: x / num_batches, grads_decoder)
    state_decoder = state_decoder.apply_gradients(grads=grads_decoder)
    return state_decoder


@partial(jax.jit, static_argnames=("lossfn",))
def update_state_batch_em(
    key: chex.ArrayDevice,
    X_batch: chex.ArrayDevice,
    state_batch_encoder: TrainState,
    state_decoder: TrainState,
    num_e_steps: int,
    lossfn: Callable,
):
    """
    Update the train state using the EM algorithm.
    """
    apply_fn = state_batch_encoder.apply_fn
    def part_lossfn(params_encoder, params_decoder, X):
        params = freeze({
            "params": {
                "encoder": params_encoder,
                "decoder": params_decoder
            }
        })
        return lossfn(key, params, apply_fn, X)

    # E-step
    part_e_step = partial(
        e_step,
        lossfn=part_lossfn,
        X=X_batch,
        state_decoder=state_decoder,
    )
    state_batch_encoder = jax.lax.fori_loop(0, num_e_steps, part_e_step, state_batch_encoder)

    # M-step preprocessing: Accumulate gradients
    params_encoder = state_batch_encoder.params
    params_decoder = state_decoder.params
    loss_batch, grads_decoder = jax.value_and_grad(part_lossfn, 1)(params_encoder, params_decoder, X_batch)

    return state_batch_encoder, grads_decoder, loss_batch


@jax.jit
def reconstruct_opt_state(state, new_state_batch, ixs):
    mu_state = unfreeze(state.opt_state[0].mu)
    nu_state = unfreeze(state.opt_state[0].nu)

    # Update encoder optimisation-state params
    batch_mu_encoder = unfreeze(new_state_batch.opt_state[0].mu)
    mu_state = update_pytree(mu_state, batch_mu_encoder, ixs)

    batch_nu_encoder = unfreeze(new_state_batch.opt_state[0].nu)
    nu_state = update_pytree(nu_state, batch_nu_encoder, ixs)

    opt_state_update = (
        state.opt_state[0]._replace(
            mu=freeze(mu_state),
            nu=freeze(nu_state),
        ),
    ) + state.opt_state[1:]

    return opt_state_update


@jax.jit
def update_reconstruct_state(state, new_params, new_opt_state):
    new_state = TrainState(
        step=state.step + 1,
        apply_fn=state.apply_fn,
        tx=state.tx,
        params=new_params,
        opt_state=new_opt_state,
    )
    return new_state


@partial(jax.jit, static_argnames=("lossfn",))
def train_step_batch(
    key: chex.ArrayDevice,
    X: chex.ArrayDevice,
    state_encoder: TrainState,
    state_decoder: TrainState,
    ixs: chex.ArrayDevice,
    num_e_steps: int,
    lossfn: Callable,
):
    X_batch = X[ixs]
    state_encoder_batch = create_state_encoder_batch(state_encoder, ixs)
    state_encoder_batch, m_step_grads, loss_batch = update_state_batch_em(
        key, X_batch, state_encoder_batch, state_decoder, num_e_steps, lossfn
    )

    new_encoder_params = update_pytree(state_encoder.params, state_encoder_batch.params, ixs)
    new_encoder_opt_state = reconstruct_opt_state(state_encoder, state_encoder_batch, ixs)
    new_state_encoder = update_reconstruct_state(state_encoder, new_encoder_params, new_encoder_opt_state)
    return new_state_encoder, m_step_grads, loss_batch


@jax.jit
def accumulate_grads(all_grads, grads):
    all_grads = jax.tree_map(lambda x, y: x + y, all_grads, grads)
    return all_grads


def train_epoch(
    key: jnp.ndarray,
    X: jnp.ndarray,
    state_encoder: TrainState,
    state_decoder: TrainState,
    batch_size: int,
    num_e_steps: int,
    num_m_steps: int,
    lossfn: Callable,
):
    num_obs = X.shape[0]
    key_batch, key_train = jax.random.split(key)
    batch_ixs = hlax.training.get_batch_train_ixs(key_batch, num_obs, batch_size)
    num_batches = len(batch_ixs)
    keys_train = jax.random.split(key_train, num_batches)

    total_loss = 0
    m_grads = jax.tree_map(lambda x: jnp.zeros_like(x), state_decoder.params)
    for batch_ix, key_epoch in zip(batch_ixs, keys_train):
        state_encoder, m_step_grads, loss_batch = train_step_batch(
            key_epoch, X, state_encoder, state_decoder, batch_ix, num_e_steps, lossfn
        )
        total_loss += loss_batch
        if num_m_steps > 0:
            m_grads = accumulate_grads(m_grads, m_step_grads)
    state_decoder = m_step(state_decoder, m_grads, num_batches)
    return total_loss, state_encoder, state_decoder


def train_checkpoints(
    key: chex.ArrayDevice,
    model: nn.Module,
    config: CheckpointsConfig,
    X: chex.ArrayDevice,
    lossfn: Callable,
    tx_encoder: optax.GradientTransformation,
    tx_decoder: optax.GradientTransformation,
):
    """
    Find inference model parameters theta at multiple
    epochs.
    """
    dict_params = {}
    dict_times = {}
    hist_loss = []
    num_obs, *dim_obs = X.shape

    key_params_init, key_eps_init, key_train = jax.random.split(key, 3)
    keys_train = jax.random.split(key_train, config.num_epochs)

    batch_init = jnp.ones((num_obs, *dim_obs))
    params_init = model.init(key_params_init, batch_init, key_eps_init, num_samples=3)

    params_encoder_init = params_init["params"]["encoder"]
    params_decoder_init = params_init["params"]["decoder"]

    state_encoder = TrainState.create(
        apply_fn=model.apply,
        params=params_encoder_init,
        tx=tx_encoder,
    )

    state_decoder = TrainState.create(
        apply_fn=model.apply,
        params=params_decoder_init,
        tx=tx_decoder,
    )

    num_e_steps = config.num_e_steps
    num_m_steps = 1
    time_init = time()
    for e, keyt in (pbar := tqdm(enumerate(keys_train), total=config.num_epochs)):
        loss, state_encoder, state_decoder = train_epoch(
            keyt, X, state_encoder, state_decoder, config.batch_size,
            num_e_steps, num_m_steps, lossfn
        )

        hist_loss.append(loss)
        pbar.set_description(f"loss={loss:0.5e}")

        if (enum := e + 1) in config.eval_epochs:
            time_ellapsed = time() - time_init
            params_decoder = state_decoder.params

            dict_times[f"e{enum}"] = time_ellapsed
            dict_params[f"e{enum}"] = params_decoder

    output = {
        "times": dict_times,
        "checkpoint_params": dict_params,
        "hist_loss": jnp.array(hist_loss),
        "state_final": (state_encoder, state_decoder)
    }
    return output
