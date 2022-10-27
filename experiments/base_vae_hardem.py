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
import distrax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from dataclasses import dataclass
from flax.core import freeze, unfreeze
from tqdm.auto import tqdm
from flax.training.train_state import TrainState


def neg_iwmll(key, params_encoder, params_decoder, observation,
                      encoder, decoder, num_is_samples=10):
    """
    Importance-weighted marginal log-likelihood for an unamortised, uncoditional
    gaussian encoder.
    """
    latent_samples, (mu_z, std_z) = encoder.apply(
        params_encoder, key, num_samples=num_is_samples
    )

    _, dim_latent = latent_samples.shape
    # log p(x|z)
    mu_x, logvar_x = decoder.apply(params_decoder, latent_samples)
    std_x = jnp.exp(logvar_x / 2)
    log_px_cond = distrax.MultivariateNormalDiag(mu_x, std_x).log_prob(observation)
    
    # log p(z)
    mu_z_init, std_z_init = jnp.zeros(dim_latent), jnp.ones(dim_latent)
    log_pz = distrax.MultivariateNormalDiag(mu_z_init, std_z_init).log_prob(latent_samples)
    
    # log q(z)
    log_qz = distrax.MultivariateNormalDiag(mu_z, std_z).log_prob(latent_samples)
    
    # Importance-weighted marginal log-likelihood
    log_prob = log_pz + log_px_cond - log_qz
    niwmll = -jax.nn.logsumexp(log_prob, axis=-1, b=1/num_is_samples)
    return niwmll


grad_neg_iwmll_encoder = jax.value_and_grad(neg_iwmll, argnums=1)

vmap_neg_iwmll = jax.vmap(neg_iwmll, (0, 0, None, 0, None, None, None))


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


def load_dataset(n_train, n_test):
    train, test = hlax.datasets.load_fashion_mnist(n_train, n_test)
    X_train, X_test = train[0], test[0]
    return X_train, X_test


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
        loss, state = hlax.vae.train_epoch(keyt, state, X, config.batch_size, hlax.losses.iwae)

        hist_loss.append(loss)        
        pbar.set_description(f"{loss=:.3e}")
        
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
    
    output = {
        "checkpoint_params": dict_params,
        "hist_loss": jnp.array(hist_loss),
    }
    return output


def warmup_phase(key, X_train, config_vae, config_hardem):
    key_vae, key_hardem = jax.random.split(key)

    # Obtain inference model parameters
    output_vae = warmup_vae(key_vae, config_vae, X_train)
    output_hardem = warmup_hardem(key_hardem, config_hardem, X_train)

    output =  {
        "vae": {
            **output_vae,
        },
        "hardem": {
            **output_hardem,
        },
    }

    return output


def test_single(key, config_test, output, X):
    _, dim_obs = X.shape
    key_train, key_eval= jax.random.split(key)
    keys_eval = jax.random.split(key_eval, len(X))

    encoder_test = config_test.class_encoder(config_test.dim_latent)
    decoder_test = config_test.class_decoder(dim_obs, config_test.dim_latent)

    checkpoint_vals = output["checkpoint_params"].keys()
    dict_mll_epochs = {}
    for keyv in tqdm(checkpoint_vals):
        params_decoder = output["checkpoint_params"][keyv]
        res = hlax.training.train_encoder(key_train, X, encoder_test, decoder_test,
                                          params_decoder, config_test.tx, config_test.num_epochs,
                                          grad_neg_iwmll_encoder, config_test.num_is_samples,
                                          leave=False)
        mll_values = -vmap_neg_iwmll(keys_eval, res["params"], params_decoder, X, encoder_test, decoder_test, 50)
        dict_mll_epochs[keyv] = mll_values
    return dict_mll_epochs


def test_phase(key, X_test, config_test, output_warmup):
    output_vae = output_warmup["vae"]
    output_hardem = output_warmup["hardem"]

    dict_mll_epochs = {}
    dict_mll_epochs_vae = test_single(key, config_test, output_vae, X_test)
    dict_mll_epochs_hardem = test_single(key, config_test, output_hardem, X_test)

    for keyv in dict_mll_epochs_vae.keys():
        mll_vals_vae = dict_mll_epochs_vae[keyv]
        mll_vals_hardem = dict_mll_epochs_hardem[keyv]
        mll_vals = np.c_[mll_vals_hardem, mll_vals_vae]
        dict_mll_epochs[keyv] = mll_vals
    
    return dict_mll_epochs


def main(config, dict_models):
    num_train = config["warmup"]["num_obs"]
    num_test = config["test"]["num_obs"]

    key = jax.random.PRNGKey(314)
    key_warmup, key_eval = jax.random.split(key)

    train, test = hlax.datasets.load_fashion_mnist(num_train, num_test)
    X_train, X_test = train[0], test[0]

    config_vae, config_hardem, config_test = setup(config, dict_models)

    print("Warmup phase")
    warmup_output = warmup_phase(key_warmup, X_train, config_vae, config_hardem)
    print("Test phase")
    test_output = test_phase(key_eval, X_test, config_test, warmup_output)

    output = {
        "warmup": warmup_output,
        "test": test_output,
    }

    return output
