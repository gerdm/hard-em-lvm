import os
import jax
import hlax
import sys
import tomli
import pickle
import base_vae_hardem
import numpy as np
import flax.linen as nn
from datetime import datetime, timezone
from typing import Sequence, Tuple

os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_VISIBLE_DEVICES"] = "2"

class Decoder(nn.Module):
    """
    Based on:
    https://github.com/probml/probml-utils/blob/main/probml_utils/conv_vae_flax_utils.py
    """
    dim_obs: Tuple[int, int, int] # (H)eight, (W)idth, number of (C)hannels
    dim_latent: int = 20
    hidden_channels: Sequence[int] = (32, 64, 128, 256, 512)

    @nn.compact
    def __call__(self, X, training=True):
        H, W, C = self.dim_obs

        # TODO: relax this restriction
        factor = 2 ** len(self.hidden_channels)
        assert(
            H % factor == W % factor == 0
        ), f"output_dim must be a multiple of {factor}"
        H, W = H // factor, W // factor

        X = nn.Dense(H * W * self.hidden_channels[-1])(X)
        X = nn.elu(X)
        X = X.reshape((-1, H, W, self.hidden_channels[-1]))

        for hidden_channel in reversed(self.hidden_channels[:-1]):
            X = nn.ConvTranspose(
                hidden_channel, (3, 3), strides=(2, 2), padding=((1, 2), (1, 2))
            )(X)
            # X = nn.BatchNorm(use_running_average=not training)(X)
            X = nn.elu(X)

        X = nn.ConvTranspose(C, (3, 3), strides=(2, 2), padding=((1, 2), (1, 2)))(X)
        return X


class Encoder(nn.Module):
    latent_dim: int
    hidden_channels: Sequence[int] = (32, 64, 128, 256, 512)

    @nn.compact
    def __call__(self, X, training=True):
        for channel in self.hidden_channels:
            X = nn.Conv(channel, (3, 3), strides=2, padding=1)(X)
            # X = nn.BatchNorm(use_running_average=not training)(X)
            X = nn.elu(X)

        X = X.reshape((-1, np.prod(X.shape[-3:])))
        mu = nn.Dense(self.latent_dim)(X)
        logvar = nn.Dense(self.latent_dim)(X)

        return mu, logvar


class ConvEncoder(nn.Module):
    latent_dim: tuple

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
    dim_obs: Tuple
    dim_latent: int

    @nn.compact
    def __call__(self, z):
        x = nn.Dense(28 ** 2)(z)
        x = x.reshape(*z.shape[:-1], *self.dim_obs)
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
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    path_config = "./experiments/configs/fmnistv02.toml"
    name_file = sys.argv[0]
    with open(path_config, "rb") as f:
        config = tomli.load(f)

    num_warmup = config["warmup"]["num_obs"]
    num_test = config["test"]["num_obs"]
    warmup, test = hlax.datasets.load_fashion_mnist(num_warmup, num_test, melt=False)
    X_warmup, X_test = warmup[0], test[0]

    # # Resizing to 32x32
    # X_warmup = jax.vmap(jax.image.resize, (0, None, None))(X_warmup, (32, 32), "nearest")
    # X_test = jax.vmap(jax.image.resize, (0, None, None))(X_test, (32, 32), "nearest")
    # # Reshaping to 32x32x1
    X_warmup = X_warmup[..., None]
    X_test = X_test[..., None]

    key = jax.random.PRNGKey(314)
    lossfn_vae = hlax.losses.iwae_bern
    lossfn_hardem = hlax.losses.loss_hard_nmll

    _, *dim_obs = X_warmup.shape
    dim_latent = config["setup"]["dim_latent"]
    model_vae = hlax.models.VAEBern(dim_latent, dim_obs, ConvEncoder, ConvDecoder)
    model_decoder = ConvDecoder(dim_obs, dim_latent)
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
    )


    output["metadata"] = {
        "config": config,
        "timestamp": now,
        "name_file": name_file,
        "path_config": path_config,
    }

    print(f"Saving {now}")
    with open(f"./experiments/outputs/experiment-{now}.pkl", "wb") as f:
        pickle.dump(output, f)
