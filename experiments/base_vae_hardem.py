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

ToDo:
*  normailsed passing initialised or unititialised encoder / decoder
"""

import jax
import hlax
import optax
import chex
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from datetime import datetime
from functools import partial
from dataclasses import dataclass
from flax.core import freeze, unfreeze
from tqdm.auto import tqdm
from flax.training.train_state import TrainState


@dataclass
class WarmupConfigVAE:
    num_epochs: int
    batch_size: int
    dim_latent: int
    eval_epochs: list

    tx_vae: optax.GradientTransformation
    num_is_samples: int

    class_decoder: nn.Module
    class_encoder: nn.Module


@dataclass
class WarmupConfigHardEM:
    num_epochs: int
    batch_size: int
    dim_latent: int
    eval_epochs: list

    num_its_params: int
    num_its_latent: int

    tx_params: optax.GradientTransformation
    tx_latent: optax.GradientTransformation

    class_decoder: nn.Module


@dataclass
class TestConfig:
    num_epochs: int
    num_is_samples: int
    dim_latent: int
    tx: optax.GradientTransformation
    class_encoder: nn.Module # Unamortised
    class_decoder: nn.Module


def setup(config, dict_models):
    # q(z|x)
    Decoder = dict_models["class_decoder"]
    # p(x|z)
    Encoder = dict_models["class_encoder"]
    EncoderTest = dict_models["class_encoder_test"]

    learning_rate = config["warmup"]["learning_rate"]
    learning_rate_test = config["warmup"]["learning_rate"]

    tx_vae = optax.adam(learning_rate)
    tx_params = optax.adam(learning_rate)
    tx_latent = optax.adam(learning_rate)
    tx_test = optax.adam(learning_rate_test)

    config_vae = WarmupConfigVAE(
        num_epochs=config["warmup"]["num_epochs"],
        batch_size=config["warmup"]["batch_size"],
        dim_latent=config["warmup"]["dim_latent"],
        eval_epochs=config["warmup"]["eval_epochs"],
        num_is_samples=config["warmup"]["vae"]["num_is_samples"],
        tx_vae=tx_vae,
        class_encoder=Encoder,
        class_decoder=Decoder,
    )
    
    config_hardem = WarmupConfigHardEM(
        num_epochs=config["warmup"]["num_epochs"],
        batch_size=config["warmup"]["batch_size"],
        dim_latent=config["warmup"]["dim_latent"],
        eval_epochs=config["warmup"]["eval_epochs"],
        num_its_params=config["warmup"]["hard_em"]["num_its_params"],
        num_its_latent=config["warmup"]["hard_em"]["num_its_latent"],
        tx_params=tx_params,
        tx_latent=tx_latent,
        class_decoder=Decoder,
    )

    config_test = TestConfig(
        num_epochs=config["test"]["num_epochs"],
        num_is_samples=config["test"]["num_is_samples"],
        dim_latent=config["warmup"]["dim_latent"],
        tx=tx_test,
        class_encoder=EncoderTest,
        class_decoder=Decoder,
    )

    return config_vae, config_hardem, config_test


def warmup_vae(
    key: chex.ArrayDevice,
    config: WarmupConfigVAE,
    X: chex.ArrayDevice
):
    """
    Find inference model parameters theta
    """
    dict_params = {}
    hist_loss = []
    _, dim_obs = X.shape

    key_params_init, key_eps_init, key_train = jax.random.split(key, 3)
    keys_train = jax.random.split(key_train, config.num_epochs)
    batch_init = jnp.ones((config.batch_size, dim_obs))

    model = hlax.models.VAE_IW(config.dim_latent, dim_obs, config.class_encoder, config.class_decoder)
    params_init = model.init(key_params_init, batch_init, key_eps_init, num_samples=3)

    state = TrainState.create(
        apply_fn=partial(model.apply, num_samples=config.num_is_samples),
        params=params_init,
        tx=config.tx_vae,
        )

    for e, keyt in (pbar := tqdm(enumerate(keys_train), total=len(keys_train))):
        loss, state = hlax.vae.train_epoch(keyt, state, X, config.batch_size)

        hist_loss.append(loss)        
        pbar.set_description(f"{loss=:.3e}")
        
        if (enum := e + 1) in config.eval_epochs:
            params_vae = state.params
            params_decoder_vae = freeze({"params": unfreeze(params_vae)["params"]["decoder"]})
            
            dict_params[f"e{enum}"] = params_decoder_vae

    return dict_params, jnp.array(hist_loss)


def warmup_hardem(
    key: chex.ArrayDevice,
    config: WarmupConfigHardEM,
    X: chex.ArrayDevice
):
    """
    Find inference model parameters theta
    using the Hard EM algorithm
    """
    dict_params = {}
    hist_loss = []
    _, dim_obs = X.shape
    decoder = config.class_decoder(dim_obs, config.dim_latent)
    lossfn = hlax.hard_decoder.loss_hard_nmll

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
        nll, params_decoder, z_est, opt_states = res
        hist_loss.append(nll)
        pbar.set_description(f"{nll=:.3e}")
        
        if (enum := e + 1) in config.eval_epochs:
            dict_params[f"e{enum}"] = params_decoder
    return dict_params, jnp.array(hist_loss)


def load_dataset(n_train, n_test):
    train, test = hlax.datasets.load_fashion_mnist(n_train, n_test)
    X_train, X_test = train[0], test[0]
    return X_train, X_test


def warmup_phase(key, X_train, config_vae, config_hardem):
    key_vae, key_hardem = jax.random.split(key)

    # Obtain inference model parameters
    dict_params_vae, hist_loss_vae = warmup_vae(key_vae, config_vae, X_train)
    dict_params_hardem, hist_loss_hardem = warmup_hardem(key_hardem, config_hardem, X_train)

    output =  {
        "vae": {
            "checkpoint_params": dict_params_vae,
            "hist_loss": hist_loss_vae,
        },
        "hardem": {
            "checkpoint_params": dict_params_hardem,
            "hist_loss": hist_loss_hardem,
        },
    }

    return output


def test_phase(key, X_test, config_test, output_warmup):
    _, dim_obs = X_test.shape

    encoder_test = config_test.class_encoder(config_test.dim_latent)
    decoder_test = config_test.class_decoder(dim_obs, config_test.dim_latent)

    key_train, key_eval= jax.random.split(key)
    keys_eval = jax.random.split(key_eval, len(X_test))

    vmap_neg_iwmll = jax.vmap(hlax.training.neg_iwmll, (0, 0, None, 0, None, None, None))

    dict_mll_epochs = {}

    checkpoint_vals = output_warmup["hardem"]["checkpoint_params"].keys()
    for keyv in tqdm(checkpoint_vals):
        params_decoder_vae = output_warmup["vae"]["checkpoint_params"][keyv]
        params_decoder_hem = output_warmup["hardem"]["checkpoint_params"][keyv]


        vae_res = hlax.training.train_encoder(key_train, X_test, encoder_test, decoder_test,
                                            params_decoder_vae, config_test.tx, config_test.num_epochs,
                                            config_test.num_is_samples, leave=False)

        hem_res = hlax.training.train_encoder(key_train, X_test, encoder_test, decoder_test,
                                            params_decoder_hem, config_test.tx, config_test.num_epochs,
                                            config_test.num_is_samples, leave=False)
        
        mll_vae = -vmap_neg_iwmll(keys_eval, vae_res["params"], params_decoder_vae, X_test, encoder_test, decoder_test, 50)
        mll_hem = -vmap_neg_iwmll(keys_eval, hem_res["params"], params_decoder_hem, X_test, encoder_test, decoder_test, 50)
        
        mll_vals = np.c_[mll_hem, mll_vae]
        dict_mll_epochs[keyv] = mll_vals 
    
    return dict_mll_epochs


def main(config, dict_models):
    num_train = config["warmup"]["num_obs"]
    num_test = config["test"]["num_obs"]

    key = jax.random.PRNGKey(314)
    key_warmup, key_eval = jax.random.split(key)

    train, test = hlax.datasets.load_fashion_mnist(num_train, num_test)
    X_train, X_test = train[0], test[0]

    config_vae, config_hardem, config_test  = setup(config, dict_models)

    print("Warmup phase")
    warmup_output = warmup_phase(key_warmup, X_train, config_vae, config_hardem)
    print("Test phase")
    test_output = test_phase(key_eval, X_test, config_test, warmup_output)

    output = {
        "warmup": warmup_output,
        "test": test_output,
    }

    return output


if __name__ == "__main__":
    import os
    import sys
    import tomli
    import pickle
    from datetime import datetime, timezone

    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
    os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
    os.environ["TPU_VISIBLE_DEVICES"] = "1"

    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    path_config = sys.argv[1]
    with open(path_config, "rb") as f:
        config = tomli.load(f)

    dict_models = {
        "class_decoder": hlax.models.DiagDecoder,
        "class_encoder": hlax.models.EncoderSimple,
        "class_encoder_test": hlax.models.GaussEncoder,
    }

    output = main(config, dict_models)

    output["metadata"] = {
        "config": config,
        "timestamp": now,
    }

    with open(f"./experiments/outputs/experiment-{now}.pkl", "wb") as f:
        pickle.dump(output, f)