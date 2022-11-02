"""
In this experiment, we consider a latent variable model
        p(x, z) = p(x|z) p(z)
Where p(x|z) is parametrized by a neural network with
parameters theta and p(z) is a standard Gaussian.

We estimate the parameters theta of the inference model
1. Defining a variational distribution q(z|x; phi) using
    the IWAE estimator.
2. Using the Hard EM algorithm to estimate the parameters
    theta of the inference model directly.

After estimating theta, we consider a variational
distribution q(zn|xn; phi{n}) and estimate the parameters phi{n}
for each n=1,...,N using the IWAE estimator.

We compare the performance of the two methods by evaluating
the marginal likelihood p(x) = \int p(x|z) p(z) dz using
the importance sampling estimator.
"""

import jax
import hlax
import optax
import chex
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable
from functools import partial
from dataclasses import dataclass
from flax.core import freeze, unfreeze
from tqdm.auto import tqdm
from flax.training.train_state import TrainState


@dataclass
class WarmupConfigVAE:
    model_vae: nn.Module

    num_epochs: int
    batch_size: int
    dim_latent: int
    eval_epochs: list

    tx_vae: optax.GradientTransformation
    num_is_samples: int


@dataclass
class WarmupConfigHardEM:
    model_decoder: nn.Module

    num_epochs: int
    batch_size: int
    dim_latent: int
    eval_epochs: list

    num_its_params: int
    num_its_latent: int

    tx_params: optax.GradientTransformation
    tx_latent: optax.GradientTransformation


@dataclass
class TestConfig:
    num_epochs: int
    num_is_samples: int
    dim_latent: int
    tx: optax.GradientTransformation
    model_encoder: nn.Module # Unamortised
    model_decoder: nn.Module


def load_dataset(n_train, n_test):
    train, test = hlax.datasets.load_fashion_mnist(n_train, n_test)
    X_train, X_test = train[0], test[0]
    return X_train, X_test


def setup(config, model_vae, model_decoder, model_encoder_test):
    learning_rate = config["warmup"]["learning_rate"]
    learning_rate_test = config["warmup"]["learning_rate"]

    tx_vae = optax.adam(learning_rate)
    tx_params = optax.adam(learning_rate)
    tx_latent = optax.adam(learning_rate)
    tx_test = optax.adam(learning_rate_test)

    config_vae = WarmupConfigVAE(
        num_epochs=config["warmup"]["num_epochs"],
        batch_size=config["warmup"]["batch_size"],
        dim_latent=config["setup"]["dim_latent"],
        eval_epochs=config["warmup"]["eval_epochs"],
        num_is_samples=config["warmup"]["vae"]["num_is_samples"],
        tx_vae=tx_vae,
        model_vae=model_vae,
    )

    config_hardem = WarmupConfigHardEM(
        num_epochs=config["warmup"]["num_epochs"],
        batch_size=config["warmup"]["batch_size"],
        dim_latent=config["setup"]["dim_latent"],
        eval_epochs=config["warmup"]["eval_epochs"],
        num_its_params=config["warmup"]["hard_em"]["num_its_params"],
        num_its_latent=config["warmup"]["hard_em"]["num_its_latent"],
        tx_params=tx_params,
        tx_latent=tx_latent,
        model_decoder=model_decoder,
    )

    config_test = TestConfig(
        num_epochs=config["test"]["num_epochs"],
        num_is_samples=config["test"]["num_is_samples"],
        dim_latent=config["setup"]["dim_latent"],
        tx=tx_test,
        model_encoder=model_encoder_test,
        model_decoder=model_decoder,
    )

    return config_vae, config_hardem, config_test


def warmup_vae(
    key: chex.ArrayDevice,
    config: WarmupConfigVAE,
    X: chex.ArrayDevice,
    lossfn: Callable,
):
    """
    Find inference model parameters theta
    """
    dict_params = {}
    hist_loss = []
    _, *dim_obs = X.shape

    key_params_init, key_eps_init, key_train = jax.random.split(key, 3)
    keys_train = jax.random.split(key_train, config.num_epochs)
    batch_init = jnp.ones((config.batch_size, *dim_obs))

    model = config.model_vae
    params_init = model.init(key_params_init, batch_init, key_eps_init, num_samples=3)

    state = TrainState.create(
        apply_fn=partial(model.apply, num_samples=config.num_is_samples),
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


def warmup_hardem(
    key: chex.ArrayDevice,
    config: WarmupConfigHardEM,
    X: chex.ArrayDevice,
    lossfn: Callable,
):
    """
    Find inference model parameters theta
    using the Hard EM algorithm
    """
    dict_params = {}
    hist_loss = []
    decoder = config.model_decoder

    key_init, key_step = jax.random.split(key)
    keys_step = jax.random.split(key_step, config.num_epochs)

    states = hlax.hard_decoder.initialise_state(
        key_init,
        decoder,
        config.tx_params,
        config.tx_latent,
        X,
        config.dim_latent,
    )
    opt_states, target_states = states
    params_decoder, z_est = target_states

    pbar = tqdm(enumerate(keys_step), total=config.num_epochs)
    for e, keyt in pbar:
        res = hlax.hard_decoder.train_epoch_adam(
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
            dict_params[f"e{enum}"] = params_decoder

    output = {
        "checkpoint_params": dict_params,
        "hist_loss": jnp.array(hist_loss),
    }
    return output


def warmup_phase(key, X_train, config_vae, config_hardem, lossfn_vae, lossfn_hardem):
    key_vae, key_hardem = jax.random.split(key)

    # Obtain inference model parameters
    output_hardem = warmup_hardem(key_hardem, config_hardem, X_train, lossfn_hardem)
    output_vae = warmup_vae(key_vae, config_vae, X_train, lossfn_vae)

    output =  {
        "vae": {
            **output_vae,
        },
        "hardem": {
            **output_hardem,
        },
    }

    return output


def test_single(key, config_test, output, X, grad_loss_encoder, vmap_loss_encoder, num_is_samples=50):
    key_train, key_eval = jax.random.split(key)
    keys_eval = jax.random.split(key_eval, len(X))

    encoder_test = config_test.model_encoder
    decoder_test = config_test.model_decoder

    checkpoint_vals = output["checkpoint_params"].keys()
    dict_mll_epochs = {}
    for keyv in tqdm(checkpoint_vals):
        params_decoder = output["checkpoint_params"][keyv]
        res = hlax.training.train_encoder(key_train, X, encoder_test, decoder_test,
                                          params_decoder, config_test.tx, config_test.num_epochs,
                                          grad_loss_encoder, config_test.num_is_samples,
                                          leave=False)
        mll_values = -vmap_loss_encoder(keys_eval, res["params"], params_decoder, X, encoder_test, decoder_test, num_is_samples)
        dict_mll_epochs[keyv] = mll_values
    return dict_mll_epochs


def test_phase(key, X_test, config_test, output_warmup, grad_loss_encoder, vmap_loss_encoder):
    output_vae = output_warmup["vae"]
    output_hardem = output_warmup["hardem"]

    dict_mll_epochs = {}
    dict_mll_epochs_vae = test_single(key, config_test, output_vae, X_test, grad_loss_encoder, vmap_loss_encoder)
    dict_mll_epochs_hardem = test_single(key, config_test, output_hardem, X_test, grad_loss_encoder, vmap_loss_encoder)

    for keyv in dict_mll_epochs_vae.keys():
        mll_vals_vae = dict_mll_epochs_vae[keyv]
        mll_vals_hardem = dict_mll_epochs_hardem[keyv]
        mll_vals = np.c_[mll_vals_hardem, mll_vals_vae]
        dict_mll_epochs[keyv] = mll_vals

    return dict_mll_epochs


def main(
    key: chex.ArrayDevice,
    X_train: chex.ArrayDevice,
    X_test: chex.ArrayDevice,
    config: dict,
    model_vae: nn.Module,
    model_decoder: nn.Module,
    model_encoder_test: nn.Module,
    lossfn_vae: Callable,
    lossfn_hardem: Callable,
    grad_loss_encoder: Callable,
    vmap_loss_encoder: Callable,
):
    key_warmup, key_eval = jax.random.split(key)
    config_vae, config_hardem, config_test = setup(config, model_vae, model_decoder, model_encoder_test)

    print("Warmup phase")
    warmup_output = warmup_phase(key_warmup, X_train, config_vae, config_hardem, lossfn_vae, lossfn_hardem)
    print("Test phase")
    test_output = test_phase(key_eval, X_test, config_test, warmup_output, grad_loss_encoder, vmap_loss_encoder)

    output = {
        "warmup": warmup_output,
        "test": test_output,
    }

    return output
