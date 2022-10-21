import hlax
import os
import sys
import tomli
import pickle
import base_vae_hardem
import flax.linen as nn
from datetime import datetime, timezone

os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_VISIBLE_DEVICES"] = "1"


class ConvEncoder(nn.Module):
    latent_dim: tuple
    
    @nn.compact
    def __call__(self, x):
        z = nn.Conv(5, (3, 3), padding="SAME")(x)
        z = nn.elu(z)
        z = nn.max_pool(z, (2, 2), padding="SAME")        
        z = z.reshape((z.shape[0], -1))
        z = nn.Dense(self.latent_dim)(z)
        return z
    

class ConvDecoder(nn.Module):
    observed_dims: tuple

    def setup(self):
        self.activation = nn.elu
        self.mean = nn.Dense(self.dim_full, use_bias=True, name="mean")
        self.logvar = nn.Dense(self.dim_full, use_bias=False, name="logvar")
    
    @nn.compact
    def __call__(self, z):
        x = nn.Dense(28 ** 2)(z)
        x = x.reshape(-1, 28, 28, 1)
        x = self.activation(x)
        x = nn.Conv(5, (3, 3), padding="SAME")(x)
        x = self.activation(x)
        x = nn.Conv(1, (3, 3), padding="SAME")(x)
        return x


class Decoder(nn.Module):
    """
    For the generative model
    p(x,z) = p(x|z) * p(z)
    """
    dim_full: int
    dim_latent: int = 20
    
    def setup(self):
        self.activation = nn.elu
        self.mean = nn.Dense(self.dim_full, use_bias=True, name="mean")
        self.logvar = nn.Dense(self.dim_full, use_bias=False, name="logvar")
    
    @nn.compact
    def __call__(self, z):
        x = nn.Dense(100)(z)
        x = self.activation(x)
        x = nn.Dense(100)(x)
        x = self.activation(x)
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
        z = nn.relu(z)
        z = nn.Dense(self.n_hidden)(z)
        z = nn.relu(z)
        z = nn.Dense(self.n_hidden)(z)
        z = nn.relu(z)
        mean_z = nn.Dense(self.latent_dim)(z)
        logvar_z = nn.Dense(self.latent_dim)(z)
        return mean_z, logvar_z


if __name__ == "__main__":
    import sys
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    path_config = "./experiments/configs/fmnistv02.toml"
    name_file = sys.argv[0]
    with open(path_config, "rb") as f:
        config = tomli.load(f)

    dict_models = {
        "class_decoder": Decoder,
        "class_encoder": Encoder,   
        "class_encoder_test": hlax.models.GaussEncoder,
    }

    output = base_vae_hardem.main(config, dict_models)

    output["metadata"] = {
        "config": config,
        "timestamp": now,
        "name_file": name_file,
        "path_config": path_config,
    }

    print(f"Saving {now}")
    with open(f"./experiments/outputs/experiment-{now}.pkl", "wb") as f:
        pickle.dump(output, f)


