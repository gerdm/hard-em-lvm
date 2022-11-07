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
from typing import Callable, Dict, Union
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


def train_test(
    key: chex.ArrayDevice,
    X_train: chex.ArrayDevice,
    X_test: chex.ArrayDevice,
    config_train: Union[hlax.hard_em_lvm.CheckpointsConfig,
                        hlax.vae.CheckpointsConfig],
    config_test: TestConfig,
    lossfn_train: Callable,
    vmap_loss_encoder_test: Callable,
    grad_loss_encoder_test: Callable,
    train_checkpoints: Callable
) -> Dict:
    key_train, key_test = jax.random.split(key)
    output_train = train_checkpoints(key_train, config_train, X_train, lossfn_train)
    output_test = test_decoder_params(key_test, config_test, output_train, X_test, grad_loss_encoder_test, vmap_loss_encoder_test)
    
    res = {
        "train": output_train,
        "test": output_test,
    }
    return res


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
    key_hardem, key_vae = jax.random.split(key)
    config_vae, config_hardem, config_test = setup(config, model_vae, model_decoder, model_encoder_test)

    print("Hard EM")
    res_hemlvm = train_test(key_hardem, X_train, X_test, config_hardem, config_test,
                            lossfn_hardem, vmap_loss_encoder, grad_loss_encoder, hlax.hard_em_lvm.train_checkpoints)
    print("VAE")
    res_vae = train_test(key_vae, X_train, X_test, config_vae, config_test,
                         lossfn_vae, vmap_loss_encoder, grad_loss_encoder, hlax.vae.train_checkpoints)

    output = {
        "hardem": res_hemlvm,
        "vae": res_vae,
    }

    return output
