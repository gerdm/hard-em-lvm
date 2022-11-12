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
import chex
import flax.linen as nn
from typing import Callable, Dict, Union
from tqdm.auto import tqdm


def setup(config, model_vae, model_decoder, model_test):
    config_vae = hlax.vae.load_config(config, model_vae)
    config_hardem = hlax.hard_em_lvm.load_config(config, model_decoder)
    config_test = hlax.unamortised.load_test_config(config, model_test, model_decoder)

    return config_vae, config_hardem, config_test


def test_decoder_checkpoints(
    key: chex.ArrayDevice,
    X: chex.ArrayDevice,
    config_test: hlax.unamortised.Config,
    checkpoint_params: Dict,
    lossfn: Callable,
) -> Dict:
    checkpoint_vals = checkpoint_params.keys()
    num_checkpoints = len(checkpoint_vals)
    keys = jax.random.split(key, num_checkpoints)

    dict_params = {}
    dict_losses = {}
    pbar = tqdm(zip(keys, checkpoint_vals), total=num_checkpoints)
    for key_checkpoint, epoch_name in pbar:
        params_checkpoint = checkpoint_params[epoch_name]["params"]
        res_checkpoint = hlax.unamortised.test_decoder(key_checkpoint, config_test, X, lossfn, params_checkpoint)
        params_test = res_checkpoint["state"].params
        hist_loss = res_checkpoint["hist_loss"]

        dict_params[epoch_name] = params_test
        dict_losses[epoch_name] = hist_loss
    
    res = {
        "mll_epochs": dict_losses,
        "params": dict_params,
    }
    return res


def train_test(
    key: chex.ArrayDevice,
    X_train: chex.ArrayDevice,
    X_test: chex.ArrayDevice,
    config_train: Union[hlax.unamortised.CheckpointsConfig,
                        hlax.vae.CheckpointsConfig],
    config_test: hlax.unamortised.Config,
    lossfn_train: Callable,
    lossfn_test: Callable,
    train_checkpoints: Callable
) -> Dict:
    key_train, key_test = jax.random.split(key)
    output_train = train_checkpoints(key_train, config_train, X_train, lossfn_train)

    checkpoint_params = output_train["checkpoint_params"]
    output_test = test_decoder_checkpoints(key_test, X_test, config_test, checkpoint_params, lossfn_test)
    
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
