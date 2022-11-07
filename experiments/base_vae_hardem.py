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
import flax.linen as nn
from typing import Callable, Dict
from dataclasses import dataclass
from tqdm.auto import tqdm


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

def load_test_config(
    config: Dict,
    model_encoder: nn.Module,
    model_decoder: nn.Module,
) -> TestConfig:
    """
    Load the test configuration.
    """
    learning_rate = config["test"]["learning_rate"]
    tx_test = optax.adam(learning_rate)

    pconfig = TestConfig(
        num_epochs=config["test"]["num_epochs"],
        num_is_samples=config["test"]["num_is_samples"],
        dim_latent=config["setup"]["dim_latent"],
        tx=tx_test,
        model_encoder=model_encoder,
        model_decoder=model_decoder,
    )
    return pconfig


def setup(config, model_vae, model_decoder, model_encoder_test):
    config_vae = hlax.vae.load_config(config, model_vae)
    config_hardem = hlax.hard_em_lvm.load_config(config, model_decoder)
    config_test = load_test_config(config, model_encoder_test, model_decoder)

    return config_vae, config_hardem, config_test


def warmup_phase(key, X_train, config_vae, config_hardem, lossfn_vae, lossfn_hardem):
    key_vae, key_hardem = jax.random.split(key)

    # Obtain inference model parameters
    output_hardem = hlax.hard_em_lvm.train_checkpoints(key_hardem, config_hardem, X_train, lossfn_hardem)
    output_vae = hlax.vae.train_checkpoints(key_vae, config_vae, X_train, lossfn_vae)

    output =  {
        "vae": {
            **output_vae,
        },
        "hardem": {
            **output_hardem,
        },
    }

    return output


def test_decoder_params(key, config_test, output, X, grad_loss_encoder, vmap_loss_encoder, num_is_samples=50):
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
    dict_mll_epochs_vae = test_decoder_params(key, config_test, output_vae, X_test, grad_loss_encoder, vmap_loss_encoder)
    dict_mll_epochs_hardem = test_decoder_params(key, config_test, output_hardem, X_test, grad_loss_encoder, vmap_loss_encoder)

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
