import jax
import distrax
import jax.numpy as jnp


def iwae(key, params, apply_fn, X_batch):
    """
    Loss function
    Importance-weighted Variational Autoencoder (IW-VAE)
    with Gaussian Decoder
    """
    batch_size = len(X_batch)
    keys = jax.random.split(key, batch_size)

    encode_decode = jax.vmap(apply_fn, (None, 0, 0))
    encode_decode = encode_decode(params, X_batch, keys)
    z, (mean_z, logvar_z), (mean_x, logvar_x) = encode_decode
    _, num_is_samples, dim_latent = z.shape

    std_z = jnp.exp(logvar_z / 2)
    std_x = jnp.exp(logvar_x / 2)

    dist_prior = distrax.MultivariateNormalDiag(jnp.zeros(dim_latent),
                                                jnp.ones(dim_latent))
    dist_decoder = distrax.MultivariateNormalDiag(mean_x, std_x)
    dist_posterior = distrax.Normal(mean_z[:, None, :], std_z[:, None, :])

    log_prob_z_prior = dist_prior.log_prob(z)
    log_prob_x = dist_decoder.log_prob(X_batch[:, None, :])
    log_prob_z_post = dist_posterior.log_prob(z).sum(axis=-1)

    log_prob = log_prob_z_prior + log_prob_x - log_prob_z_post

    # negative Importance-weighted marginal log-likelihood
    niwmll = -jax.nn.logsumexp(log_prob, axis=-1, b=1/num_is_samples).mean()
    return niwmll


def loss_hard_nmll(params, z_batch, X_batch, model):
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
        Decoder model (input z, output x)
    """
    dim_latent = model.dim_latent

    mean_x, logvar_x = model.apply(params, z_batch)
    std_x = jnp.exp(logvar_x / 2)

    dist_prior = distrax.MultivariateNormalDiag(jnp.zeros(dim_latent), jnp.ones(dim_latent))
    dist_decoder = distrax.MultivariateNormalDiag(mean_x, std_x)

    log_prob_z_prior = dist_prior.log_prob(z_batch)
    log_prob_x = dist_decoder.log_prob(X_batch)

    log_prob = log_prob_z_prior + log_prob_x

    return -log_prob.mean()


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
