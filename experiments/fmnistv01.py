import jax
import hlax
import sys
import tomli
import pickle
import base_vae_hardem
import flax.linen as nn
from datetime import datetime, timezone

class Decoder(nn.Module):
    """
    For the generative model
    p(x,z) = p(x|z) * p(z)
    """
    dim_full: int
    dim_latent: int = 20
    
    def setup(self):
        self.mean = nn.Dense(self.dim_full, use_bias=True, name="mean")
        self.logvar = nn.Dense(self.dim_full, use_bias=False, name="logvar")
    
    @nn.compact
    def __call__(self, z):
        x = nn.Dense(20)(z)
        x = nn.elu(x)
        mean_x = self.mean(x)
        logvar_x = self.logvar(x)
        return mean_x, logvar_x 


class Encoder(nn.Module):
    """
    two-layered encoder
    """
    latent_dim: int
    n_hidden: int = 100
    
    @nn.compact
    def __call__(self, x):
        z = nn.Dense(self.n_hidden)(x)
        z = nn.elu(z)
        z = nn.Dense(self.n_hidden)(z)
        z = nn.elu(z)
        mean_z = nn.Dense(self.latent_dim)(z)
        logvar_z = nn.Dense(self.latent_dim)(z)
        return mean_z, logvar_z


if __name__ == "__main__":
    import os
    import sys

    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
    os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
    os.environ["TPU_VISIBLE_DEVICES"] = "1"

    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    name_file, *path_config = sys.argv
    path_config = "./experiments/configs/fmnistv01.toml"

    with open(path_config, "rb") as f:
        config = tomli.load(f)

    num_warmup = config["warmup"]["num_obs"]
    num_test = config["test"]["num_obs"]
    warmup, test = hlax.datasets.load_fashion_mnist(num_warmup, num_test)
    X_warmup, X_test = warmup[0], test[0]

    key = jax.random.PRNGKey(314)
    lossfn_vae = hlax.losses.iwae
    lossfn_hardem = hlax.losses.loss_hard_nmll

    grad_neg_iwmll_encoder = jax.value_and_grad(hlax.losses.neg_iwmll_bern, argnums=1)
    vmap_neg_iwmll = jax.vmap(hlax.losses.neg_iwmll_bern, (0, 0, None, 0, None, None, None))

    _, dim_obs = X_warmup.shape
    dim_latent = config["setup"]["dim_latent"]
    model_vae = hlax.models.VAEGauss(dim_latent, dim_obs, Encoder, Decoder)
    model_decoder = Decoder(dim_obs, dim_latent)
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
        "path_config": path_config,
    }

    print(f"Saving {now}")
    with open(f"./experiments/outputs/experiment-{now}-mlp.pkl", "wb") as f:
        pickle.dump(output, f)
