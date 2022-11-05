import os
import jax
import hlax
import sys
import tomli
import pickle
import base_vae_hardem
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


class ConvVAE(nn.Module):
    observed_dims: tuple
    latent_dim: int

    def setup(self):
        self.encoder = ConvEncoder(self.latent_dim)
        self.decoder = ConvDecoder(self.observed_dims)

    def encode(self, x):
        return self.encoder(x)

    def __call__(self, x):
        return self.decoder(self.encoder(x))


if __name__ == "__main__":
    import sys

    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
    os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
    os.environ["TPU_VISIBLE_DEVICES"] = "1"

    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    path_config = "./experiments/configs/fmnist-conv01.toml"
    name_file = sys.argv[0]
    with open(path_config, "rb") as f:
        config = tomli.load(f)

    num_warmup = config["warmup"]["num_obs"]
    num_test = config["test"]["num_obs"]
    warmup, test = hlax.datasets.load_fashion_mnist(num_warmup, num_test, melt=False, normalize=False)
    X_warmup, X_test = warmup[0], test[0]

    X_warmup = X_warmup[..., None]
    X_test = X_test[..., None]

    key = jax.random.PRNGKey(314)
    lossfn_vae = hlax.losses.iwae_bern
    lossfn_hardem = hlax.losses.hard_nmll_bern

    grad_neg_iwmll_encoder = jax.value_and_grad(hlax.losses.neg_iwmll_bern, argnums=1)
    vmap_neg_iwmll = jax.vmap(hlax.losses.neg_iwmll_bern, (0, 0, None, 0, None, None, None))

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
        "path_config": path_config,
    }

    print(f"Saving {now}")
    with open(f"./experiments/outputs/experiment-{now}-conv.pkl", "wb") as f:
        pickle.dump(output, f)
