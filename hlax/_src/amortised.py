import jax
import hlax
import chex
import optax
import flax.linen as nn
import jax.numpy as jnp
from time import time
from tqdm.auto import tqdm
from typing import Callable, Dict
from functools import partial
from flax.core import freeze, unfreeze
from dataclasses import dataclass
from flax.training.train_state import TrainState


@dataclass
class TrainConfig:
    num_epochs: int
    batch_size: int


@dataclass
class CheckpointsConfig(TrainConfig):
    eval_epochs: list


def load_config(
    dict_config: Dict,
    model: nn.Module,
) -> CheckpointsConfig:
    """
    Load config from dict.
    """
    learning_rate = dict_config["train"]["learning_rate"]

    tx = optax.adam(learning_rate)

    config = hlax.vae.CheckpointsConfig(
        num_epochs=dict_config["train"]["num_epochs"],
        batch_size=dict_config["train"]["batch_size"],
        dim_latent=dict_config["setup"]["dim_latent"],
        eval_epochs=dict_config["train"]["eval_epochs"],
        num_is_samples=dict_config["train"]["vae"]["num_is_samples"],
        tx=tx,
        model=model,
    )
    return config


@partial(jax.jit, static_argnames="lossfn")
def train_step(state, X, key, lossfn):
    params = state.params
    apply_fn = state.apply_fn
    
    loss, grads = jax.value_and_grad(lossfn, 1)(key, params, apply_fn, X)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


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


@partial(jax.jit, static_argnames="lossfn")
def train_step_encoder(state, X, key, lossfn):
    """
    Perform an E-step update for a generative model.
    """
    apply_fn = state.apply_fn
    def part_lossfn(params_encoder, params_decoder, X):
        params = freeze({
            "params": {
                "encoder": params_encoder,
                "decoder": params_decoder
            }
        })
        return lossfn(key, params, apply_fn, X)

    # E-step only
    params_fixed = "decoder"
    params_decoder = state.params["params"]["decoder"]
    params_encoder = state.params["params"]["encoder"]
    grads_zero_decoder = jax.tree_map(lambda _: 0, params_decoder)
    old_mu_params = state.opt_state[0].mu["params"][params_fixed]
    old_nu_params = state.opt_state[0].nu["params"][params_fixed]
    loss, grads_encoder = jax.value_and_grad(part_lossfn, 0)(params_encoder, params_decoder, X)
    grads_patch = freeze({
        "params": {
            "encoder": grads_encoder,
            "decoder": grads_zero_decoder
        }
    })
    state = state.apply_gradients(grads=grads_patch)
    state = adam_replace_opt_state(
        state,
        old_mu_params,
        old_nu_params,
        "decoder"
    )

    return loss, state


def train_epoch_encoder(key, state, X, batch_size, lossfn):
    num_samples = len(X)
    key_batch, keys_vae = jax.random.split(key)
    batch_ixs = hlax.training.get_batch_train_ixs(key_batch, num_samples, batch_size)
    
    num_batches = len(batch_ixs)
    keys_vae = jax.random.split(keys_vae, num_batches)
    
    total_loss = 0
    for key_vae, batch_ix in zip(keys_vae, batch_ixs):
        X_batch = hlax.training.index_values_batch(X, batch_ix)
        loss, state = train_step_encoder(state, X_batch, key_vae, lossfn)
        total_loss += loss
    
    return total_loss.item(), state


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


def train_encoder(
    key: chex.ArrayDevice,
    config: TrainConfig,
    X: chex.ArrayDevice,
    state: TrainState,
    lossfn: Callable,
):
    """
    Train the inference network of a latent variable model.
    We fix the parameters of the generative model (decoder)
    and only update the parameters of the inference model (encoder).
    """
    hist_loss = []
    keys_train = jax.random.split(key, config.num_epochs)
    pbar = tqdm(keys_train)
    for keyt in pbar:
        loss, state = train_epoch_encoder(keyt, state, X, config.batch_size, lossfn)
        hist_loss.append(loss)
        pbar.set_description(f"generative-{loss=:.3e}")
    return state, hist_loss


def train_checkpoints(
    key: chex.ArrayDevice,
    config: CheckpointsConfig,
    X: chex.ArrayDevice,
    lossfn: Callable,
):
    """
    Find inference model parameters theta at multiple epochs.
    """
    dict_params = {}
    dict_times = {}
    hist_loss = []
    _, *dim_obs = X.shape

    key_params_init, key_eps_init, key_train = jax.random.split(key, 3)
    keys_train = jax.random.split(key_train, config.num_epochs)
    batch_init = jnp.ones((config.batch_size, *dim_obs))

    params_init = config.model.init(key_params_init, batch_init, key_eps_init, num_samples=3)

    state = TrainState.create(
        apply_fn=partial(config.model.apply, num_samples=config.num_is_samples),
        params=params_init,
        tx=config.tx,
        )

    time_init = time()
    for e, keyt in (pbar := tqdm(enumerate(keys_train), total=len(keys_train))):
        loss, state = train_epoch(keyt, state, X, config.batch_size, lossfn)

        hist_loss.append(loss)
        pbar.set_description(f"vae-{loss=:.3e}")

        if (enum := e + 1) in config.eval_epochs:
            time_ellapsed = time() - time_init
            params_vae = state.params
            params_decoder_vae = freeze({"params": unfreeze(params_vae)["params"]["decoder"]})

            dict_times[f"e{enum}"] = time_ellapsed
            dict_params[f"e{enum}"] = params_decoder_vae

    output = {
        "times": dict_times,
        "checkpoint_params": dict_params,
        "hist_loss": jnp.array(hist_loss),
        "state_final": state,
    }
    return output
