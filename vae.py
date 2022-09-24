import jax
import einops
import distrax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Callable
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

class IPL:
    def __init__(self, prior_mu, prior_Sigma, decoder, n_iterations=1):
        self.prior_mu = prior_mu
        self.prior_Sigma = prior_Sigma
        self.decoder = decoder
        self.num_iterations = n_iterations
        self.dim_obs = decoder.dim_full
        self.dim_latent = decoder.dim_latent
        self.num_is_samples = 13
        
    @partial(jax.jit, static_argnums=(0,))
    def lin_predict(self, mu, Sigma, params):
        m, logPsi = self.decoder.apply(params, mu)
        G = jax.jacfwd(lambda x: self.decoder.apply(params, x)[0])(mu)
        Psi = jnp.exp(logPsi) * jnp.eye(self.dim_obs)
        
        S = jnp.einsum("im,mk,jk->ij", G, Sigma, G)
        S = S + Psi
        C = Sigma @ G.T
        
        return m, S, C

    
    def gauss_condition(self, mu_prev, Sigma_prev, y, m, S, C):
        K = jnp.linalg.solve(S, C.T).T
        mu_est = mu_prev + K @ (y - m)

        Sigma_est = jnp.einsum("im,mk,jk->ij", K, S, K)
        Sigma_est = Sigma_prev - Sigma_est

        return mu_est, Sigma_est
    
    
    def iterate_posterior_step(self, state, y, params):
        """
        Iterate posterior step
        """
        mu, Sigma = state
        m, S, C = self.lin_predict(mu, Sigma, params)
        mu_est, Sigma_est = self.gauss_condition(mu, Sigma, y, m, S, C)
        new_state = (mu_est, Sigma_est)
        return new_state, None
    
    
    def estimate_posterior_params(self, y, params):
        state = (self.prior_mu, self.prior_Sigma)
        ips = partial(self.iterate_posterior_step, params=params)
        
        # lax.scan makes it painfully slow
        for it in range(self.num_iterations):
            state, _ = ips(state, y)
            
        post_mu, post_Sigma = state
        return post_mu, post_Sigma
    

    @partial(jax.jit, static_argnums=(0,))
    def compute_iwlmm_single(self, key, obs, params):
        mu, Sigma = self.estimate_posterior_params(obs, params)
        dist_posterior_latent = distrax.MultivariateNormalFullCovariance(mu, Sigma)
        dist_prior_latent = distrax.MultivariateNormalFullCovariance(self.prior_mu, self.prior_Sigma)

        is_samples = dist_posterior_latent.sample(seed=key, sample_shape=self.num_is_samples)
        
        mean_x, logvar_x = self.decoder.apply(params, is_samples)
        cov_x = jax.vmap(jnp.diag)(jnp.exp(logvar_x / 2))
        # cov_x = jnp.exp(logvar_x) * jnp.eye(self.dim_obs)
        
        dist_decoder = distrax.MultivariateNormalFullCovariance(mean_x, cov_x)

        log_is = (dist_decoder.log_prob(obs)
                + dist_prior_latent.log_prob(is_samples)
                - dist_posterior_latent.log_prob(is_samples))
        
        return jax.nn.logsumexp(log_is, b=1/self.num_is_samples)


class Encoder(nn.Module):
    """
    For the inference model p(z|x)
    """
    latent_dim: int
    n_hidden: int = 5
    
    @nn.compact
    def __call__(self, x):
        z = nn.Dense(self.n_hidden)(x)
        z = nn.relu(z)
        mean_z = nn.Dense(self.latent_dim)(z)
        logvar_z = nn.Dense(self.latent_dim)(z)
        return mean_z, logvar_z


@partial(jax.jit, static_argnames='dim')
def fill_lower_tri(v, dim):
    """
    Fill a vector with lower triangular (without diagonal)
    values into a square matrix.

    Source: https://github.com/google/jax/discussions/10146
    """
    idx = jnp.tril_indices(dim, k=-1)
    return jnp.zeros((dim, dim), dtype=v.dtype).at[idx].set(v)


class EncoderFullCov(nn.Module):
    """
    For the inference model p(z|x)
    """
    latent_dim: int
    n_hidden: int = 5

    def setup(self):
        init_tri = nn.initializers.normal(stddev=1e-4)
        # Number of elments in the lower (without diagonal) triangular matrix
        tril_dim = self.latent_dim * (self.latent_dim + 1) // 2 - self.latent_dim
        self.hidden_layer = nn.Dense(self.n_hidden, name="latent_hidden")
        self.mean_layer = nn.Dense(self.latent_dim, name="latent_mean")
        self.logvardiag_layer = nn.Dense(self.latent_dim, use_bias=False, name="latent_logvardiag", kernel_init=init_tri)
        self.tril_layer  = nn.Dense(tril_dim, name="latent_tril", use_bias=False, kernel_init=init_tri)


    @nn.compact
    def __call__(self, x):
        z = self.hidden_layer(x)
        z = nn.relu(z)

        mean_z = self.mean_layer(z)
        logvar_z = self.logvardiag_layer(z)
        diag_z = jax.vmap(jnp.diag)(jnp.exp(logvar_z / 2))
        Lz = self.tril_layer(z)
        Lz = jax.vmap(fill_lower_tri, (0, None))(Lz, self.latent_dim)

        return mean_z, Lz + diag_z


class Decoder(nn.Module):
    """
    Parameterise the generative model
    p(x,z) = p(x|z) * p(z)
    """
    dim_full: int
    dim_latent: int
    normal_init: Callable = nn.initializers.normal()
    uniform_init: Callable = nn.initializers.uniform()
        
    def setup(self):
        self.b = self.param("b", self.normal_init, (self.dim_full,))
        self.A = self.param("A", self.normal_init, (self.dim_full, self.dim_latent))
        self.logPhi = self.param("logPsi", self.normal_init, (1,))
    
    def __call__(self, z):
        mean_x = jnp.einsum("...m,dm->...d", z, self.A)+ self.b
        logvar_x = self.logPhi
        
        return mean_x, logvar_x
    
    
class VAE(nn.Module):
    latent_dim: int
    full_dim: int
    n_hidden: int = 5
    
    @staticmethod
    def reparameterise(key, mean, logvar):
        std = jnp.exp(logvar / 2)
        eps = jax.random.normal(key, logvar.shape)
        z = mean + eps * std
        return z
    
    def setup(self):
        self.encoder = Encoder(self.latent_dim, self.n_hidden)
        self.decoder = Decoder(self.full_dim, self.latent_dim)
    
    def __call__(self, x, key_eps):
        mean_z, logvar_z = self.encoder(x)
        z = VAE.reparameterise(key_eps, mean_z, logvar_z)
        mean_x, logvar_x = self.decoder(z)
        return (mean_z, logvar_z), (mean_x, logvar_x)
    

class VAEIW(nn.Module):
    """
    Importance-Weighted Variational Autoencoder
    """
    latent_dim: int
    full_dim: int
    n_hidden: int = 5
    
    @staticmethod
    def reparameterise(key, mean, logvar, num_samples=1):
        std = jnp.exp(logvar / 2)
        eps = jax.random.normal(key, (num_samples, *logvar.shape))
        z = mean[None, ...] + jnp.einsum("...d,...d->...d", std, eps)
        return z
    
    def setup(self):
        self.encoder = Encoder(self.latent_dim, self.n_hidden)
        self.decoder = Decoder(self.full_dim, self.latent_dim)
    
    def __call__(self, x, key_eps, num_samples=1):
        mean_z, logvar_z = self.encoder(x)
        z = VAEIW.reparameterise(key_eps, mean_z, logvar_z, num_samples)
        mean_x, logvar_x = self.decoder(z)
        return z, (mean_z, logvar_z), (mean_x, logvar_x)


class VAEIWFD(nn.Module):
    """
    Importance-Weighted Variational Autoencoder with Full Covariance Matrix
    for the posterior distribution
    """
    latent_dim: int
    full_dim: int
    n_hidden: int = 5
    
    def reparameterise(self, key, mean, cov, num_samples):
        num_obs = len(mean)
        eps = jax.random.normal(key, (num_samples, num_obs, self.latent_dim))
        z = mean + jnp.einsum("...dm,...m->...d", cov, eps)
        return z
    
    def setup(self):
        self.encoder = EncoderFullCov(self.latent_dim, self.n_hidden)
        self.decoder = Decoder(self.full_dim, self.latent_dim)
    
    def __call__(self, x, key_eps, num_samples=1):
        mean_z, L_z = self.encoder(x)
        z = self.reparameterise(key_eps, mean_z, L_z, num_samples)
        mean_x, logvar_x = self.decoder(z)
        return z, (mean_z, L_z), (mean_x, logvar_x)


def sgvb(params, model, X_batch, key):
    """
    Loss function
    -------------
    
    Stochastic Gradient Variational Bayes (SGVB)
    estimator.
    """
    batch_size = len(X_batch)
    keys = jax.random.split(key, batch_size)
    
    encode_decode = jax.vmap(model.apply, (None, 0, 0))
    encode_decode = encode_decode(params, X_batch, keys)
    (mean_z, logvar_z), (mean_x, logvar_x) = encode_decode
    
    kl = (-logvar_z + jnp.exp(logvar_z) + mean_z ** 2 - 1) / 2
    
    std_x = jnp.exp(logvar_x / 2)
    probx = distrax.Normal(mean_x, std_x)
    log_probs_x = probx.log_prob(X_batch)
    
    elbo_est = log_probs_x.sum(axis=-1) - kl.sum(axis=-1)
    return -elbo_est.mean()


def iwae(params, model, X_batch, key, num_is_samples=13):
    """
    Loss function
    -------------
    
    Importance-weight Variational Autoencoder (IW-VAE)
    """
    batch_size = len(X_batch)
    keys = jax.random.split(key, batch_size)
    
    encode_decode = jax.vmap(model.apply, (None, 0, 0, None))
    encode_decode = encode_decode(params, X_batch, keys, num_is_samples)
    z, (mean_z, logvar_z), (mean_x, logvar_x) = encode_decode
    std_z = jnp.exp(logvar_z / 2)
    std_x = jnp.exp(logvar_x / 2)
    
    dist_prior = distrax.MultivariateNormalDiag(jnp.zeros(model.latent_dim), jnp.ones(model.latent_dim))
    dist_decoder = distrax.MultivariateNormalDiag(mean_x, std_x[..., None] * jnp.ones(model.full_dim))
    dist_posterior = distrax.Normal(mean_z[:, None, :], std_z[:, None, :])
    
    log_prob_z_prior = dist_prior.log_prob(z)
    log_prob_x = dist_decoder.log_prob(X_batch[:, None, :])
    log_prob_z_post = dist_posterior.log_prob(z).sum(axis=-1)
    
    log_prob = log_prob_z_prior + log_prob_x - log_prob_z_post
    
    # negative Importance-weighted marginal log-likelihood
    niwmll = -jax.nn.logsumexp(log_prob, axis=-1, b=1/num_is_samples).sum()
    return niwmll
    

def iwae_fullcov(params, model, X_batch, key, num_is_samples=13):
    """
    Loss function
    -------------
    
    Importance-weight Variational Autoencoder (IW-VAE)
    """
    batch_size = len(X_batch)
    # keys = jax.random.split(key, batch_size)
    
    # encode_decode = jax.vmap(model.apply, (None, None, 0, None))
    encode_decode = model.apply(params, X_batch, key, num_is_samples)
    z, (mean_z, L_z), (mean_x, logvar_x) = encode_decode
    std_x = jnp.exp(logvar_x / 2)

    dist_prior = tfd.MultivariateNormalDiag(jnp.zeros(model.latent_dim), jnp.ones(model.latent_dim))
    dist_decoder = tfd.MultivariateNormalDiag(mean_x, std_x * jnp.ones(model.full_dim))
    dist_posterior = tfd.MultivariateNormalTriL(mean_z, L_z)

    log_prob_z_prior = dist_prior.log_prob(z)
    log_prob_x = dist_decoder.log_prob(X_batch)
    log_prob_z_post = dist_posterior.log_prob(z)
    
    log_prob = log_prob_z_prior + log_prob_x - log_prob_z_post
    
    # negative Importance-weighted marginal log-likelihood
    niwmll = -jax.nn.logsumexp(log_prob, axis=0, b=1/num_is_samples).sum()
    return niwmll
    

def get_batch_train_ixs(key, num_samples, batch_size):
    """
    Obtain the training indices to be used in an epoch of
    mini-batch optimisation.
    """
    steps_per_epoch = num_samples // batch_size
    
    batch_ixs = jax.random.permutation(key, num_samples)
    batch_ixs = batch_ixs[:steps_per_epoch * batch_size]
    batch_ixs = batch_ixs.reshape(steps_per_epoch, batch_size)
    
    return batch_ixs


def train_step(state, X_batch, key, model, loss_fn, **kwargs):
    loss_fn = partial(loss_fn,
                      model=model,
                      X_batch=X_batch,
                      key=key,
                      **kwargs)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


def estimate_mll(state, observations):
    """
    Estimate the marginal log-likelihood of a
    model of the form:
    z ~ N(z; 0, I)
    x ~ N(x; Az + b, r * I)
    """
    params = state.params["params"]["decoder"]
    dim_full, *_ = params["b"].shape

    Psi = jnp.eye(dim_full) * jnp.exp(params["logPsi"])
    marginal_pdf_est = distrax.MultivariateNormalFullCovariance(
        loc=params["b"],
        covariance_matrix=params["A"] @ params["A"].T + Psi
    )

    log_likelihood_est = marginal_pdf_est.log_prob(observations).sum()
    return log_likelihood_est
