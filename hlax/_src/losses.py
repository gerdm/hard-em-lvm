import jax
import distrax
import jax.numpy as jnp


def iwae(key, params, apply_fn, X_batch):
    """
    Loss function
    Importance-weighted Variational Autoencoder (IW-VAE)
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
