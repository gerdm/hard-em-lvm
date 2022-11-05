import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable


class FADecoder(nn.Module):
    """
    Factor Analysis (FA) model
    Parameterise the generative model
    p(x,z) = p(x|z) * p(z)
    """
    dim_obs: int
    dim_latent: int
    normal_init: Callable = nn.initializers.normal()
    uniform_init: Callable = nn.initializers.uniform()

    def setup(self):
        self.b = self.param("b", self.normal_init, (self.dim_obs,))
        self.A = self.param("A", self.normal_init, (self.dim_obs, self.dim_latent))
        # self.logPsi = self.param("logPsi", self.normal_init, (self.dim_obs, self.dim_latent))
        self.logPsi = self.param("logPsi", self.normal_init, (self.dim_obs,))

    def eval_mean(self, z):
        mean_x = jnp.einsum("...m,dm->...d", z, self.A)+ self.b
        return mean_x

    def eval_diag_cov(self, z):
        # logvar_x = jnp.einsum("...m,dm->...d", z, self.logPsi)
        zeros = jnp.zeros((self.dim_obs, self.dim_latent))
        logvar_x = jnp.einsum("...m,dm->...d", z, zeros) + self.logPsi

        return logvar_x

    def __call__(self, z):
        mean_x = self.eval_mean(z)
        logvar_x = self.eval_diag_cov(z)

        return mean_x, logvar_x


class HomkDecoder(nn.Module):
    """
    Parameterise the generative model
    p(x,z) = p(x|z) * p(z)
    as a homoskedastic generative process
    """
    dim_obs: int
    dim_latent: int = 20
    normal_init: Callable = nn.initializers.normal()
    activation = nn.relu

    def setup(self):
        self.logPsi = self.param("logPsi", self.normal_init, (self.dim_obs,))

    def eval_diag_cov(self, z):
        # logvar_x = jnp.einsum("...m,dm->...d", z, self.logPsi)
        zeros = jnp.zeros((self.dim_obs, self.dim_latent))
        logvar_x = jnp.einsum("...m,dm->...d", z, zeros) + self.logPsi

        return logvar_x

    @nn.compact
    def __call__(self, z):
        x = nn.Dense(30)(z)
        x = self.activation(x)
        mean_x = nn.Dense(self.dim_obs, use_bias=True)(x)
        logvar_x = self.eval_diag_cov(z)

        return mean_x, logvar_x


class DiagDecoder(nn.Module):
    """
    For the generative model
    p(x,z) = p(x|z) * p(z)
    """
    dim_full: int
    dim_latent: int = 20

    def setup(self):
        self.activation = nn.tanh
        self.hidden = nn.Dense(20, name="hidden")
        self.mean = nn.Dense(self.dim_full, use_bias=True, name="mean")
        self.logvar = nn.Dense(self.dim_full, use_bias=False, name="logvar")

    def __call__(self, z):
        x = self.hidden(z)
        x = self.activation(x)
        mean_x = self.mean(x)
        logvar_x = self.logvar(x)
        return mean_x, logvar_x


class EncoderFullCov(nn.Module):
    """
    For the inference model p(z|x)
    """
    latent_dim: int
    n_hidden: int = 5

    def setup(self):
        init_tri = nn.initializers.normal(stddev=1e-5)
        # Number of elments in the lower (without diagonal) triangular matrix
        tril_dim = self.latent_dim * (self.latent_dim + 1) // 2 - self.latent_dim
        self.mean_layer = nn.Dense(self.latent_dim, name="latent_mean")
        self.logvardiag_layer = nn.Dense(self.latent_dim, use_bias=False, name="latent_logvardiag", kernel_init=init_tri)
        self.tril_layer  = nn.Dense(tril_dim, name="latent_tril", use_bias=False, kernel_init=init_tri)

    @nn.compact
    def __call__(self, x):
        raise NotImplementedError("Not implemented yet")
        z = nn.Dense(self.n_hidden, name="latent_hiddent_1")(x)
        z = nn.relu(z)

        mean_z = self.mean_layer(z)
        logvar_z = self.logvardiag_layer(z)

        diag_z = jax.vmap(jnp.diag)(jnp.exp(logvar_z / 2))
        # diag_z = jnp.diag(jnp.exp(logvar_z / 2))
        Lz = self.tril_layer(z)
        Lz = jax.vmap(vae.fill_lower_tri, (0, None))(Lz, self.latent_dim)

        return mean_z, Lz + diag_z


class GaussEncoder(nn.Module):
    dim_latent: int
    normal_init: Callable = nn.initializers.normal()

    def setup(self):
        self.mu = self.param("mu", self.normal_init, (self.dim_latent,))
        self.logvar_diag = self.param("logvar_diag", self.normal_init, (self.dim_latent,))

    def sample_proposal(self, key, num_samples):
        std = jnp.exp(self.logvar_diag / 2)
        eps = jax.random.normal(key, (num_samples, self.dim_latent))
        z = self.mu[None, ...] + jnp.einsum("d,...d->...d", std, eps)
        return z

    def __call__(self, key, num_samples=1):
        std = jnp.exp(self.logvar_diag / 2)
        z_samples = self.sample_proposal(key, num_samples=num_samples)

        return z_samples, (self.mu, std)


class EncoderSimple(nn.Module):
    """
    two-layered encoder
    """
    latent_dim: int
    n_hidden: int = 5

    @nn.compact
    def __call__(self, x):
        z = nn.Dense(self.n_hidden)(x)
        z = nn.relu(z)
        z = nn.Dense(self.n_hidden)(z)
        z = nn.relu(z)
        mean_z = nn.Dense(self.latent_dim)(z)
        logvar_z = nn.Dense(self.latent_dim)(z)
        return mean_z, logvar_z


class VAEGauss(nn.Module):
    """
    Base class for variational autoencoder with
    gaussian decoder p(x|z) and gaussian encoder p(z|x)

    # TODO: set decoder and decoder to initialied models
    """
    latent_dim: int
    obs_dim: int
    Encoder: nn.Module
    Decoder: nn.Module

    def reparameterise(self, key, mean, logvar, num_samples=1):
        std = jnp.exp(logvar / 2)
        eps = jax.random.normal(key, (num_samples, *logvar.shape))
        z = mean[None, ...] + jnp.einsum("...d,...d->...d", std, eps)
        return z

    def setup(self):
        self.encoder = self.Encoder(self.latent_dim)
        self.decoder = self.Decoder(self.obs_dim, self.latent_dim)

    def __call__(self, x, key_eps, num_samples=1):
        mean_z, logvar_z = self.encoder(x)
        z = self.reparameterise(key_eps, mean_z, logvar_z, num_samples)
        mean_x, logvar_x = self.decoder(z)
        return z, (mean_z, logvar_z), (mean_x, logvar_x)


class VAEBern(nn.Module):
    """
    Base class for a variational autoencoder
    with Bernoulli decoder p(x|z) = Bern(x| f(z))
    and Gaussian encoder q(z|x) = N(z|mu(x), sigma(x))

    # TODO: set decoder and encoder to initialied models
    """
    latent_dim: int
    obs_dim: int
    Encoder: nn.Module
    Decoder: nn.Module

    def reparameterise(self, key, mean, logvar, num_samples=1):
        std = jnp.exp(logvar / 2)
        eps = jax.random.normal(key, (num_samples, *logvar.shape))
        z = mean[None, ...] + jnp.einsum("...d,...d->...d", std, eps)
        return z

    def setup(self):
        self.encoder = self.Encoder(self.latent_dim)
        self.decoder = self.Decoder(self.obs_dim, self.latent_dim)

    def __call__(self, x, key_eps, num_samples=1):
        mean_z, logvar_z = self.encoder(x)
        z = self.reparameterise(key_eps, mean_z, logvar_z, num_samples)
        logit_mean_x = self.decoder(z)
        return z, (mean_z, logvar_z), logit_mean_x


class UnamortisedVAEBern(nn.Module):
    """
    Base class for an unamortised variational autoencoder
    with Bernoulli decoder p(x|z) = Bern(x| f(z))
    and Gaussian encoder q(z|x) = N(z|mu(x), sigma(x))
    """
    latent_dim: int
    obs_dim: int
    Encoder: nn.Module
    Decoder: nn.Module

    def reparameterise(self, key, mean, logvar, num_samples=1):
        std = jnp.exp(logvar / 2)
        eps = jax.random.normal(key, (num_samples, *logvar.shape))
        z = mean[None, ...] + jnp.einsum("...d,...d->...d", std, eps)
        return z

    def setup(self):
        self.encoder = nn.vmap(
            self.Encoder,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=0,
        )(self.latent_dim)
        self.decoder = self.Decoder(self.obs_dim, self.latent_dim)

    def __call__(self, x, key_eps, num_samples=1):
        mean_z, logvar_z = self.encoder(x)
        z = self.reparameterise(key_eps, mean_z, logvar_z, num_samples)
        logit_mean_x = self.decoder(z)
        return z, (mean_z, logvar_z), logit_mean_x
