from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict
from torchvision import models


def atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class FullyConnectedNetwork(nn.Module):
    def __init__(self, output_dim: int, arch: str = "256-256"):
        super().__init__()
        layers: List[nn.Module] = []
        hidden_sizes = [int(h) for h in arch.split("-") if h]
        for hidden_size in hidden_sizes:
            layers.append(nn.LazyLinear(hidden_size))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.LazyLinear(output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(input_tensor)


class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type: str):
        super().__init__()
        resnet_type = resnet_type.lower()
        if resnet_type == "resnet18":
            backbone = models.resnet18(weights=None)
        elif resnet_type == "resnet34":
            backbone = models.resnet34(weights=None)
        elif resnet_type == "resnet50":
            backbone = models.resnet50(weights=None)
        elif resnet_type == "resnet101":
            backbone = models.resnet101(weights=None)
        elif resnet_type == "resnet152":
            backbone = models.resnet152(weights=None)
        elif resnet_type == "resnet200":
            backbone = models.resnet152(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNetPolicy(nn.Module):
    output_dim: int
    config_updates: ConfigDict | None

    def __init__(self, output_dim: int, config_updates: ConfigDict | None = None):
        super().__init__()
        self.output_dim = output_dim
        self.config = self.get_default_config(config_updates)
        self.share_resnet = self.config.share_resnet_between_views
        if self.share_resnet:
            self.shared_resnet = ResNetBackbone(self.config.resnet_type)
        else:
            self.resnets = nn.ModuleList()
        if self.config.state_injection in ("full", "z_only"):
            self.state_projection = nn.LazyLinear(self.config.state_projection_dim)
        else:
            self.state_projection = None
        self.mlp = FullyConnectedNetwork(self.output_dim, self.config.mlp_arch)

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.resnet_type = "ResNet18"
        config.spatial_aggregate = "average"
        config.mlp_arch = "256-256"
        config.state_injection = "full"
        config.state_projection_dim = 64
        config.share_resnet_between_views = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    @staticmethod
    def rng_keys():
        return ()

    @staticmethod
    def get_weight_decay_exclusions():
        return ("bias", "bn", "norm")

    def _get_resnet_for_view(self, index: int) -> nn.Module:
        if self.share_resnet:
            return self.shared_resnet
        if len(self.resnets) <= index:
            for _ in range(index + 1 - len(self.resnets)):
                self.resnets.append(ResNetBackbone(self.config.resnet_type))
        return self.resnets[index]

    def _process_image(self, image: torch.Tensor, index: int) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError("Images must have shape (B, H, W, C) or (B, C, H, W)")
        if image.shape[1] not in (1, 3) and image.shape[-1] in (1, 3):
            image = image.permute(0, 3, 1, 2)
        image = image.contiguous()
        resnet = self._get_resnet_for_view(index)
        resnet = resnet.to(image.device)
        return resnet(image)

    def forward(self, state: torch.Tensor, images: Sequence[torch.Tensor], return_features: bool = False):
        features: List[torch.Tensor] = []
        for i, image in enumerate(images):
            z = self._process_image(image, i)
            if self.config.spatial_aggregate == "average":
                z = z.mean(dim=(2, 3))
            elif self.config.spatial_aggregate == "flatten":
                z = z.flatten(start_dim=1)
            else:
                raise ValueError("Unsupported spatial aggregation type")
            features.append(z)

        if features:
            state = state.to(features[0].device)

        if self.config.state_injection == "full":
            if self.state_projection is None:
                raise ValueError("State projection not initialised")
            projected = self.state_projection(state)
            features.append(projected)
        elif self.config.state_injection == "z_only":
            if self.state_projection is None:
                raise ValueError("State projection not initialised")
            projected = self.state_projection(state[:, 2:3])
            features.append(projected)
        elif self.config.state_injection == "none":
            pass
        else:
            raise ValueError(f"Unsupported state_injection: {self.config.state_injection}")

        features_tensor = torch.cat(features, dim=1)
        fc_out = self.mlp(features_tensor)
        if return_features:
            return features_tensor, fc_out
        return fc_out


class TanhGaussian:
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean
        self.log_std = torch.clamp(log_std, -20.0, 2.0)
        self.std = torch.exp(self.log_std)
        self.normal = torch.distributions.Normal(self.mean, self.std)
        self.base = torch.distributions.Independent(self.normal, 1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        clipped = torch.clamp(value, -1 + eps, 1 - eps)
        pre_tanh = atanh(clipped)
        log_det = torch.sum(torch.log1p(-clipped.pow(2) + eps), dim=-1)
        base_log_prob = self.base.log_prob(pre_tanh)
        return base_log_prob - log_det

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(self.mean)
        pre_tanh = self.mean + self.std * noise
        value = torch.tanh(pre_tanh)
        log_prob = self.log_prob(value)
        return value, log_prob

    def deterministic_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        value = torch.tanh(self.mean)
        log_prob = self.log_prob(value)
        return value, log_prob


class TanhGaussianResNetPolicy(nn.Module):
    def __init__(self, output_dim: int, config_updates=None):
        super().__init__()
        self.backbone = ResNetPolicy(output_dim * 2, config_updates)

    @staticmethod
    def get_default_config(updates=None):
        return ResNetPolicy.get_default_config(updates)

    @staticmethod
    def rng_keys():
        return ()

    @staticmethod
    def get_weight_decay_exclusions():
        return ResNetPolicy.get_weight_decay_exclusions()

    def log_prob(self, state, action, images, return_mean=False):
        gaussian_params = self.backbone(state, images)
        mean, log_std = torch.chunk(gaussian_params, 2, dim=-1)
        distribution = TanhGaussian(mean, log_std)
        log_probs = distribution.log_prob(action)
        if return_mean:
            return log_probs, mean
        return log_probs

    def forward(self, state, images, deterministic=False):
        gaussian_params = self.backbone(state, images)
        mean, log_std = torch.chunk(gaussian_params, 2, dim=-1)
        distribution = TanhGaussian(mean, log_std)
        if deterministic:
            samples, log_prob = distribution.deterministic_sample()
        else:
            samples, log_prob = distribution.sample()
        return samples, log_prob


class PretrainTanhGaussianResNetPolicy(nn.Module):
    def __init__(self, output_dim: int, config_updates=None):
        super().__init__()
        self.backbone = ResNetPolicy(output_dim * 2, config_updates)

    @staticmethod
    def get_default_config(updates=None):
        return ResNetPolicy.get_default_config(updates)

    @staticmethod
    def rng_keys():
        return ()

    @staticmethod
    def get_weight_decay_exclusions():
        return ResNetPolicy.get_weight_decay_exclusions()

    def log_prob(self, state, action, images, return_mean=False):
        gaussian_params = self.backbone(state, images)
        mean, log_std = torch.chunk(gaussian_params, 2, dim=-1)
        distribution = TanhGaussian(mean, log_std)
        log_probs = distribution.log_prob(action)
        if return_mean:
            return log_probs, mean
        return log_probs

    def forward(self, state, images, deterministic=False, return_features=False):
        features, gaussian_params = self.backbone(state, images, return_features=True)
        mean, log_std = torch.chunk(gaussian_params, 2, dim=-1)
        distribution = TanhGaussian(mean, log_std)
        if deterministic:
            samples, log_prob = distribution.deterministic_sample()
        else:
            samples, log_prob = distribution.sample()
        if return_features:
            return features, samples, log_prob
        return samples, log_prob


class PrimitiveSelectionPolicy(nn.Module):
    def __init__(
        self,
        output_dim_gaussian_policy: int,
        config_updates=None,
        mlp_arch: str = "256-256",
        total_num_primitives: int = 4,
        num_embeddings: int = 5,
        num_embedding_features: int = 4,
        primitive_sequence_legnth: int = 6,
    ):
        super().__init__()
        self.output_dim_gaussian_policy = output_dim_gaussian_policy
        self.config = self.get_default_config(config_updates)
        self.embed = nn.Embedding(
            num_embeddings,
            num_embedding_features,
            padding_idx=0,
        )
        self.features_mlp = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
        )
        self.sequence_length = primitive_sequence_legnth
        self.primitive_mlp = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
        )
        self.mlp = FullyConnectedNetwork(total_num_primitives, mlp_arch)

    @staticmethod
    def get_default_config(updates=None):
        return ResNetPolicy.get_default_config(updates)

    @staticmethod
    def rng_keys():
        return ()

    @staticmethod
    def get_weight_decay_exclusions():
        return ("bias", "bn", "norm")

    def forward(self, features: torch.Tensor, primitive_sequence: torch.Tensor, deterministic: bool = False):
        if primitive_sequence.shape[-1] != self.sequence_length:
            raise ValueError("primitive_sequence has incorrect length")
        x = self.features_mlp(features)
        primitives_embeddings = self.embed(primitive_sequence.long())
        primitives_embeddings = primitives_embeddings.reshape(primitives_embeddings.shape[0], -1)
        primitives_embeddings = self.primitive_mlp(primitives_embeddings)
        embedding = torch.cat([x, primitives_embeddings], dim=-1)
        logits = self.mlp(embedding)
        return logits
