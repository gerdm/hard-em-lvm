import jax
import chex
import jax.numpy as jnp
from typing import Union


@chex.dataclass
class LinearFactorParams:
    A: chex.ArrayDevice
    b: chex.ArrayDevice
    Psi: Union[float, chex.ArrayDevice]


def sample_linear_factor_params(key, dim_obs, dim_latent, isotropic_Psi=True):
    key_A, key_b, key_Psi = jax.random.split(key, 3)
    A = jax.random.normal(key_A, (dim_obs, dim_latent))
    b = jax.random.normal(key_b, (dim_obs,))

    if isotropic_Psi:
        Psi = jax.random.uniform(key_Psi, (1,))
    else:
        Psi = jax.random.uniform(key_Psi, (dim_obs,))

    params = LinearFactorParams(A=A, b=b, Psi=Psi)
    return params


def sample_linear_factor(key, params, num_samples):
    dim_obs, dim_latent = params.A.shape
    key_latent, key_obs = jax.random.split(key)
    samples_latent = jax.random.normal(key_latent, (num_samples, dim_latent))

    samples_obs = jnp.einsum("nl,dl->nd", samples_latent, params.A) + params.b
    samples_obs = jax.random.multivariate_normal(key_obs, samples_obs, params.Psi * jnp.eye(dim_obs))

    return samples_latent, samples_obs
