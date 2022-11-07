import jax
import hlax
import chex
import optax
import flax.linen as nn
import jax.numpy as jnp
from tqdm.auto import tqdm
from typing import Callable, Dict
from functools import partial
from flax.core import freeze, unfreeze
from dataclasses import dataclass
from flax.training.train_state import TrainState


@dataclass
class CheckpointsConfig:
    model_vae: nn.Module

    num_epochs: int
    batch_size: int
    dim_latent: int
    eval_epochs: list

    tx_vae: optax.GradientTransformation
    num_is_samples: int


def load_config(
    dict_config: Dict,
    model: nn.Module,
) -> CheckpointsConfig:
    """
    Load config from dict.
    """
    learning_rate = dict_config["train"]["learning_rate"]

    tx_vae = optax.adam(learning_rate)

    config = hlax.vae.CheckpointsConfig(
        num_epochs=dict_config["train"]["num_epochs"],
        batch_size=dict_config["train"]["batch_size"],
        dim_latent=dict_config["setup"]["dim_latent"],
        eval_epochs=dict_config["train"]["eval_epochs"],
        num_is_samples=dict_config["train"]["vae"]["num_is_samples"],
        tx_vae=tx_vae,
        model_vae=model,
    )
    return config

@partial(jax.jit, static_argnames="lossfn")
def train_step(state, X, key, lossfn):
    params = state.params
    apply_fn = state.apply_fn
    
    loss, grads = jax.value_and_grad(lossfn, 1)(key, params, apply_fn, X)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


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
    hist_loss = []
    _, *dim_obs = X.shape

    key_params_init, key_eps_init, key_train = jax.random.split(key, 3)
    keys_train = jax.random.split(key_train, config.num_epochs)
    batch_init = jnp.ones((config.batch_size, *dim_obs))

    params_init = config.model_vae.init(key_params_init, batch_init, key_eps_init, num_samples=3)

    state = TrainState.create(
        apply_fn=partial(config.model_vae.apply, num_samples=config.num_is_samples),
        params=params_init,
        tx=config.tx_vae,
        )

    for e, keyt in (pbar := tqdm(enumerate(keys_train), total=len(keys_train))):
        loss, state = hlax.vae.train_epoch(keyt, state, X, config.batch_size, lossfn)

        hist_loss.append(loss)
        pbar.set_description(f"vae-{loss=:.3e}")

        if (enum := e + 1) in config.eval_epochs:
            params_vae = state.params
            params_decoder_vae = freeze({"params": unfreeze(params_vae)["params"]["decoder"]})

            dict_params[f"e{enum}"] = params_decoder_vae

    output = {
        "checkpoint_params": dict_params,
        "hist_loss": jnp.array(hist_loss),
    }
    return output
