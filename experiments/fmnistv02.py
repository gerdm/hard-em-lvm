import os
import jax
import hlax
import sys
import tomli
import pickle
import numpy as np
import base_vae_hardem
import flax.linen as nn
from datetime import datetime, timezone
from typing import Sequence, Tuple

os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_VISIBLE_DEVICES"] = "0"


class Decoder(nn.Module):
    """
    Based on:
    https://github.com/probml/probml-utils/blob/main/probml_utils/conv_vae_flax_utils.py
    """
    dim_obs: Tuple[int, int, int]
    hidden_channels: Sequence[int]
    dim_latent: int = 20

    @nn.compact
    def __call__(self, X, training):
        H, W, C = self.output_dim

        # TODO: relax this restriction
        factor = 2 ** len(self.hidden_channels)
        assert (
            H % factor == W % factor == 0
        ), f"output_dim must be a multiple of {factor}"
        H, W = H // factor, W // factor

        X = nn.Dense(H * W * self.hidden_channels[-1])(X)
        X = jax.nn.elu(X)
        X = X.reshape((-1, H, W, self.hidden_channels[-1]))

        for hidden_channel in reversed(self.hidden_channels[:-1]):
            X = nn.ConvTranspose(
                hidden_channel, (3, 3), strides=(2, 2), padding=((1, 2), (1, 2))
            )(X)
            X = nn.BatchNorm(use_running_average=not training)(X)
            X = jax.nn.elu(X)

        X = nn.ConvTranspose(C, (3, 3), strides=(2, 2), padding=((1, 2), (1, 2)))(X)
        return X


class Encoder(nn.Module):
    latent_dim: int
    hidden_channels: Sequence[int]

    @nn.compact
    def __call__(self, X, training):
        for channel in self.hidden_channels:
            X = nn.Conv(channel, (3, 3), strides=2, padding=1)(X)
            X = nn.BatchNorm(use_running_average=not training)(X)
            X = jax.nn.relu(X)

        X = X.reshape((-1, np.prod(X.shape[-3:])))
        mu = nn.Dense(self.latent_dim)(X)
        logvar = nn.Dense(self.latent_dim)(X)

        return mu, logvar


if __name__ == "__main__":
    import sys
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    path_config = "./experiments/configs/fmnistv01.toml"
    name_file = sys.argv[0]
    with open(path_config, "rb") as f:
        config = tomli.load(f)

    num_warmup = config["warmup"]["num_obs"]
    num_test = config["test"]["num_obs"]
    warmup, test = hlax.datasets.load_fashion_mnist(num_warmup, num_test, melt=False)
    X_warmup, X_test = warmup[0], test[0]

    dict_models = {
        "class_decoder": Decoder,
        "class_encoder": Encoder,   
        "class_encoder_test": hlax.models.GaussEncoder,
        "class_vae": hlax.models.VAEBern,
    }

    key = jax.random.PRNGKey(314)
    lossfn_vae = hlax.losses.iwae
    lossfn_hardem = hlax.losses.loss_hard_nmll
    output = base_vae_hardem.main(config, X_warmup, X_test, dict_models, lossfn_vae, lossfn_hardem)

    output["metadata"] = {
        "config": config,
        "timestamp": now,
        "name_file": name_file,
        "path_config": path_config,
    }

    print(f"Saving {now}")
    with open(f"./experiments/outputs/experiment-{now}.pkl", "wb") as f:
        pickle.dump(output, f)
