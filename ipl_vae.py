import jax
import distrax
import jax.numpy as jnp
from jax import lax
from functools import partial


def lin_predict(mu, Sigma, Psi, g):
    m = g(mu)
    G = jax.jacrev(g)(mu)
    
    #S = G @ Sigma @ G.T + Psi
    S = jnp.einsum("im,mk,jk->ij", G, Sigma, G)
    S = S + Psi
    C = Sigma @ G.T
    
    return (m, S, C)


def gauss_condition(mu_prev, Sigma_prev, y, m, S, C):
    dim_latent = mu_prev.shape[0]
    # K = C @ jnp.linalg.inv(S)
    K = jnp.linalg.solve(S, C.T).T
    mu_est = mu_prev + K @ (y - m)
    
    # Sigma_est = Sigma_prev - K @ S @ K.T
    Sigma_est = jnp.einsum("im,mk,jk->ij", K, S, K)
    Sigma_est = Sigma_prev - Sigma_est
    
    return mu_est, Sigma_est


def iterate_posterior_step(state, _, params, y, g):
    dim_full, *_ = y.shape
    mu, Sigma = state
    g_part = partial(g, params=params)
    Phi = jnp.eye(dim_full) * params["Psi"]
    m, S, C = partial(lin_predict, g=g_part)(mu, Sigma, Phi)
    mu_next, Sigma_next = gauss_condition(mu, Sigma, y, m, S, C)
    
    state_next = (mu_next, Sigma_next)
    return state_next, None


def compute_iwmll(key, obs_sample, params, latent, num_is_samples):
    """
    Importance weighted marginal log-likelihood (IW-MLL)
    """
    mu_latent, Sigma_latent = latent
    dim_full, *_ = obs_sample.shape
    dim_latent, *_ = mu_latent.shape

    A, b, Psi = params["A"], params["b"], params["Psi"]
    Psi = jnp.eye(dim_full) * Psi
    
    dist_posterior_latent = distrax.MultivariateNormalFullCovariance(mu_latent, Sigma_latent)
    dist_prior_latent = distrax.MultivariateNormalDiag(jnp.zeros(dim_latent), jnp.ones(dim_latent))

    is_samples = dist_posterior_latent.sample(seed=key, sample_shape=num_is_samples)

    decoder_mean = jnp.einsum("nl,dl->nd", is_samples, A) + b
    dist_decoder = distrax.MultivariateNormalFullCovariance(decoder_mean, Psi)
    
    log_is = (dist_decoder.log_prob(obs_sample)
            + dist_prior_latent.log_prob(is_samples)
            - dist_posterior_latent.log_prob(is_samples))
    
    return jax.nn.logsumexp(log_is, b=1/num_is_samples)


def estimate_posterior_latent(observations, mu_prior, Sigma_prior, g, params, iterations):
    """
    Estimate the posterior distribution a single latent vector
    """
    state_init = (mu_prior, Sigma_prior)
    part_ips = partial(iterate_posterior_step, params=params, y=observations, g=g)
    # Iterative Posterior linearization
    (mu_posterior, Sigma_posterior), _ = jax.lax.scan(part_ips, state_init, iterations)
    return mu_posterior, Sigma_posterior


def ipl_vae_step(params, xs, g, alpha, iterations, num_is_samples=50):
    mu, Sigma, y, key = xs
    grad_iwmll = jax.value_and_grad(compute_iwmll, argnums=2)
    latent_coefs = estimate_posterior_latent(y, mu, Sigma, g, params, iterations)

    ll, grads = grad_iwmll(key, y, params, latent_coefs, num_is_samples)
    params = jax.tree_map(lambda w, dw: w + alpha * dw, params, grads)
    
    return params, ll


def ipl_dvae_step(params, xs, g, alpha, iterations):
    mu, Sigma, y, key = xs
    state_init = (mu, Sigma)
    part_ips = partial(iterate_posterior_step, params=params, y=y, g=g)
    (mu_next, Sigma_next), _ = jax.lax.scan(part_ips, state_init, iterations)
    latent = mu_next, Sigma_next
    
    ll = compute_iwmll(key, y, params, latent)
    grad_iwmll = jax.grad(compute_iwmll, argnums=2)
    grads = grad_iwmll(key, y, params, latent)
    params = jax.tree_map(lambda w, dw: w + alpha * dw, params, grads)
    
    return params, (mu_next, Sigma_next, ll)


def epoch_step(params, key, mu, Sigma, ipl_step, observations):
    """
    Fix the prior mean and prior covariance matrix.
    """
    num_samples = len(observations)
    keys_is = jax.random.split(key, num_samples)
    xs = (mu, Sigma, observations, keys_is)
    params, ll = jax.lax.scan(ipl_step, params, xs)
    ll = ll.sum()
    
    return params, ll


def run_ipl_epochs(key, observations, params, mu, Sigma, alpha, n_iterations,
                   num_is_samples, num_epochs, g):
    iterations = jnp.empty(n_iterations)
    keys_epoch = jax.random.split(key, num_epochs+1)[:-1]
    part_ipl_step = partial(ipl_vae_step,
                        alpha=alpha,
                        iterations=iterations,
                        num_is_samples=num_is_samples,
                        g=g)

    part_epoch = partial(epoch_step,
                         mu=mu, Sigma=Sigma,
                         ipl_step=part_ipl_step,
                         observations=observations)
    params, hist_ll = jax.lax.scan(part_epoch, params, keys_epoch)

    return params, hist_ll


@partial(jax.vmap, in_axes=(None, 0, 0, 0, None, None))
def estimate_posterior_latent_params(params, prior_mu, prior_Sigma, y, g, n_iterations):
    state_init = (prior_mu, prior_Sigma)
    ips_iterations = jnp.empty(n_iterations)
    part_ips = partial(iterate_posterior_step, params=params, y=y, g=g)
    (posterior_mu, posterior_Sigma), _ = jax.lax.scan(part_ips, state_init, ips_iterations)
    res = {
        "mu": posterior_mu,
        "Sigma": posterior_Sigma
    }
    
    return res
