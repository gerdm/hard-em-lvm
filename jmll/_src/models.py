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
        x = nn.relu(x)
        mean_x = nn.Dense(self.dim_obs, use_bias=True)(x)
        # logvar_x = nn.Dense(self.dim_obs, use_bias=False)(x)
        logvar_x = self.eval_diag_cov(z)
        
        return mean_x, logvar_x


class DiagDecoder(nn.Module):
    """
    For the generative model
    p(x,z) = p(x|z) * p(z)
    """
    dim_full: int
    dim_latent: int = 20
    
    @nn.compact
    def __call__(self, z):
        x = nn.Dense(20)(z)
        x = nn.relu(x)
        mean_x = nn.Dense(self.dim_full)(x)
        logvar_x = nn.Dense(self.dim_full)(x)
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
        z = nn.Dense(self.n_hidden, name="latent_hiddent_1")(x)
        z = nn.relu(z)

        mean_z = self.mean_layer(z)
        logvar_z = self.logvardiag_layer(z)
        
        diag_z = jax.vmap(jnp.diag)(jnp.exp(logvar_z / 2))
        # diag_z = jnp.diag(jnp.exp(logvar_z / 2))
        Lz = self.tril_layer(z)
        Lz = jax.vmap(vae.fill_lower_tri, (0, None))(Lz, self.latent_dim)

        return mean_z, Lz + diag_z