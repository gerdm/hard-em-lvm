import os
import jax
import sys
import distrax
import hlax
import tomli
import pickle
import base_vae_hardem
import jax.numpy as jnp
import flax.linen as nn
from datetime import datetime, timezone
from typing import List


class ConvEncoder(nn.Module):
    latent_dim: List

    @nn.compact
    def __call__(self, x):
        z = nn.Conv(5, (3, 3), padding="SAME")(x)
        z = nn.elu(z)
        z = nn.max_pool(z, (2, 2), padding="SAME")
        z = z.reshape((z.shape[0], -1))
        z = nn.Dense(self.latent_dim)(z)

        mean_z = nn.Dense(self.latent_dim)(z)
        logvar_z = nn.Dense(self.latent_dim)(z)
        return mean_z, logvar_z


class ConvDecoder(nn.Module):
    dim_obs: List
    dim_latent: int

    @nn.compact
    def __call__(self, z):
        x = nn.Dense(28 ** 2)(z)
        x = x.reshape(*z.shape[:-1], *(28, 28, 1))
        x = nn.elu(x)
        x = nn.Conv(5, (3, 3), padding="SAME")(x)
        x = nn.elu(x)
        x = nn.Conv(1, (3, 3), padding="SAME")(x)
        return x


def neg_iwmll_bern(key, params_encoder, params_decoder, observation,
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
    logit_mean_x = decoder.apply(params_decoder, latent_samples)
    log_px_cond = distrax.Bernoulli(logits=logit_mean_x).log_prob(observation).sum(axis=(-1, -2, -3))

    # log p(z)
    mu_z_init, std_z_init = jnp.zeros(dim_latent), jnp.ones(dim_latent)
    log_pz = distrax.MultivariateNormalDiag(mu_z_init, std_z_init).log_prob(latent_samples)

    # log q(z)
    log_qz = distrax.MultivariateNormalDiag(mu_z, std_z).log_prob(latent_samples)

    # Importance-weighted marginal log-likelihood
    log_prob = log_pz + log_px_cond - log_qz
    niwmll = -jax.nn.logsumexp(log_prob, axis=-1, b=1/num_is_samples)
    return niwmll


def iwae_bern(key, params, apply_fn, X_batch):
    """
    Importance-weighted marginal log-likelihood for
    a Bernoulli decoder
    """
    batch_size = len(X_batch)

    # keys = jax.random.split(key, batch_size)
    # encode_decode = jax.vmap(apply_fn, (None, 0, 0))
    # encode_decode = encode_decode(params, X_batch, keys)
    encode_decode = apply_fn(params, X_batch, key)
    z, (mean_z, logvar_z), logit_mean_x = encode_decode
    _, num_is_samples, dim_latent = z.shape

    std_z = jnp.exp(logvar_z / 2)

    dist_prior = distrax.MultivariateNormalDiag(jnp.zeros(dim_latent),
                                                jnp.ones(dim_latent))
    dist_decoder = distrax.Bernoulli(logits=logit_mean_x)
    dist_posterior = distrax.Normal(mean_z[None, ...], std_z[None, ...])

    log_prob_z_prior = dist_prior.log_prob(z)
    log_prob_x = dist_decoder.log_prob(X_batch).sum(axis=(-1, -2, -3))
    log_prob_z_post = dist_posterior.log_prob(z).sum(axis=-1)

    log_prob = log_prob_z_prior + log_prob_x - log_prob_z_post

    # negative Importance-weighted marginal log-likelihood
    niwmll = -jax.nn.logsumexp(log_prob, axis=-1, b=1/num_is_samples).mean()
    return niwmll


def hard_nmll_bern(params, z_batch, X_batch, model):
    """
    Loss function
    -------------

    Negative Marginal log-likelihood for hard EM
    assuming an isotropic Gaussian prior with zero mean
    and a decoder with a diagonal covariance matrix

    Parameters
    ----------
    params: pytree
        Parameters of the decoder model, i.e.,
        model.apply(params, z_batch) = X_batch (approx)
    z_batch: jnp.ndarray
        Batch of latent variables
    X_batch: jnp.ndarray
        Batch of observations
    model: flax.nn.Module
        Decoder model (input z -> output x)
    """
    dim_latent = model.dim_latent

    logit_mean_x = model.apply(params, z_batch)

    dist_prior = distrax.MultivariateNormalDiag(jnp.zeros(dim_latent), jnp.ones(dim_latent))
    dist_decoder = distrax.Bernoulli(logits=logit_mean_x)

    log_prob_z_prior = dist_prior.log_prob(z_batch)
    log_prob_x = dist_decoder.log_prob(X_batch).sum(axis=(-1, -2, -3))

    log_prob = log_prob_z_prior + log_prob_x

    return -log_prob.mean()


if __name__ == "__main__":
    import sys

    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
    os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
    os.environ["TPU_VISIBLE_DEVICES"] = "1"

    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    name_file, *path_config = sys.argv
    if len(path_config) > 0:
        path_config = path_config[0]
    else:
        path_config = "./experiments/configs/fmnist-conv01.toml"

    with open(path_config, "rb") as f:
        config = tomli.load(f)

    num_warmup = config["warmup"]["num_obs"]
    num_test = config["test"]["num_obs"]
    warmup, test = hlax.datasets.load_fashion_mnist(num_warmup, num_test, melt=False, normalize=False)
    X_warmup, X_test = warmup[0], test[0]

    X_warmup = X_warmup[..., None]
    X_test = X_test[..., None]

    key = jax.random.PRNGKey(314)
    lossfn_vae = iwae_bern
    lossfn_hardem = hard_nmll_bern

    grad_neg_iwmll_encoder = jax.value_and_grad(neg_iwmll_bern, argnums=1)
    vmap_neg_iwmll = jax.vmap(neg_iwmll_bern, (0, 0, None, 0, None, None, None))

    _, *dim_obs = X_warmup.shape
    dim_latent = config["setup"]["dim_latent"]
    model_vae = hlax.models.VAEBern(dim_latent, dim_obs, ConvEncoder, ConvDecoder)
    model_decoder = ConvDecoder(1, dim_latent)
    model_encoder_test = hlax.models.GaussEncoder(dim_latent)

    output = base_vae_hardem.main(
        key,
        X_warmup,
        X_test,
        config,
        model_vae,
        model_decoder,
        model_encoder_test,
        lossfn_vae,
        lossfn_hardem,
        grad_neg_iwmll_encoder,
        vmap_neg_iwmll,
    )


    output["metadata"] = {
        "config": config,
        "timestamp": now,
        "name_file": name_file,
        "config": config,
    }

    print(f"Saving {now}")
    with open(f"./experiments/outputs/experiment-{now}-conv.pkl", "wb") as f:
        pickle.dump(output, f)
