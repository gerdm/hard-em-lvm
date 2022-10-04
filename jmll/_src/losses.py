import distrax
import jax.numpy as jnp


def loss_hard_nmll(params, z_batch, X_batch, model):
    """
    Loss function
    -------------
    
    Negative Marginal log-likelihood for hard EM
    """
    batch_size = len(X_batch)
    dim_latent = model.dim_latent
    
    mean_x, logvar_x = model.apply(params, z_batch)
    std_x = jnp.exp(logvar_x / 2)
    
    dist_prior = distrax.MultivariateNormalDiag(jnp.zeros(dim_latent), jnp.ones(dim_latent))
    dist_decoder = distrax.MultivariateNormalDiag(mean_x, std_x)
    
    log_prob_z_prior = dist_prior.log_prob(z_batch)
    log_prob_x = dist_decoder.log_prob(X_batch)
    
    log_prob = log_prob_z_prior + log_prob_x
    
    return -log_prob.sum()