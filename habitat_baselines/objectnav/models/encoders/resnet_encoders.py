import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from gym import spaces
from habitat import logger
from habitat_baselines.utils.common import Flatten
from habitat_baselines.rl.ddppo.policy import resnet, resnet_gn
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder


class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=128,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"depth": observation_space.spaces["depth"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint)

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        obs_depth = observations["depth"]
        if len(obs_depth.size()) == 5:
            observations["depth"] = obs_depth.contiguous().view(
                -1, obs_depth.size(2), obs_depth.size(3), obs_depth.size(4)
            )

        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)


class TorchVisionResNet50(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.
    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(
        self, observation_space, output_size, device, spatial_output: bool = False
    ):
        super().__init__()
        self.device = device
        self.resnet_layer_size = 2048
        linear_layer_input_size = 0
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            obs_size = observation_space.spaces["rgb"].shape
            obs_size_0 = observation_space.spaces["rgb"].shape[0]
            obs_size_1 = observation_space.spaces["rgb"].shape[1]
            if obs_size_0 != 224 or obs_size_1 != 224:
                logger.warn(
                    f"WARNING: TorchVisionResNet50: observation size {obs_size} is not conformant to expected ResNet input size [3x224x224]"
                )
            linear_layer_input_size += self.resnet_layer_size
        else:
            self._n_input_rgb = 0

        if self.is_blind:
            self.cnn = nn.Sequential()
            return

        self.cnn = models.resnet50(pretrained=True)

        # disable gradients for resnet, params frozen
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.cnn.fc = nn.Sequential()
            self.fc = nn.Linear(linear_layer_input_size, output_size)
            self.activation = nn.ReLU()
        else:

            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))

                    return x

            self.cnn.avgpool = SpatialAvgPool()
            self.cnn.fc = nn.Sequential()

            self.spatial_embeddings = nn.Embedding(4 * 4, 64)

            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )

        self.layer_extract = self.cnn._modules.get("avgpool")

    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """
        obs_rgb = observations["rgb"]
        if len(obs_rgb.size()) == 5:
            observations["rgb"] = obs_rgb.contiguous().view(
                -1, obs_rgb.size(2), obs_rgb.size(3), obs_rgb.size(4)
            )

        def resnet_forward(observation):
            # resnet_output = torch.zeros(1, dtype=torch.float32, device=self.device)

            # def hook(m, i, o):
            #     resnet_output.set_(o)

            # output: [BATCH x RESNET_DIM]
            # h = self.layer_extract.register_forward_hook(hook)
            # self.cnn(observation)
            # h.remove()
            # output: [BATCH x RESNET_DIM]
            resnet_output = self.cnn(observation)
            return resnet_output

        if "rgb_features" in observations:
            resnet_output = observations["rgb_features"]
        else:
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
            rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            resnet_output = resnet_forward(rgb_observations.contiguous())

        if self.spatial_output:
            b, c, h, w = resnet_output.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=resnet_output.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([resnet_output, spatial_features], dim=1)
        else:
            return self.activation(
                self.fc(torch.flatten(resnet_output, 1))
            )  # [BATCH x OUTPUT_DIM]


class ResnetRGBEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=256,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()
        backbone_split = backbone.split("_")
        logger.info("backbone: {}".format(backbone_split))
        if len(backbone_split) > 1:
            if backbone_split[1] == "gn":
                make_backbone = getattr(resnet_gn, backbone_split[0])
        else:
            make_backbone = getattr(resnet, backbone_split[0])

        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"rgb": observation_space.spaces["rgb"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=make_backbone,
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)
    
    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        obs_rgb = observations["rgb"]
        if len(obs_rgb.size()) == 5:
            observations["rgb"] = obs_rgb.contiguous().view(
                -1, obs_rgb.size(2), obs_rgb.size(3), obs_rgb.size(4)
            )

        if "rgb_features" in observations:
            x = observations["rgb_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)


class ResnetSemSeqEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=256,
        backbone="resnet18",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
        semantic_embedding_size=4,
        use_pred_semantics=False,
        use_goal_seg=False,
        is_thda=False,
    ):
        super().__init__()
        if not use_goal_seg:
            sem_input_size = 40 + 2
            self.semantic_embedder = nn.Embedding(sem_input_size, semantic_embedding_size)

        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"semantic": observation_space.spaces["semantic"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            sem_embedding_size=semantic_embedding_size,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        self.spatial_output = spatial_output
        self.use_goal_seg = use_goal_seg

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)
    
    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        obs_semantic = observations["semantic"]
        if len(obs_semantic.size()) == 5:
            observations["semantic"] = obs_semantic.contiguous().view(
                -1, obs_semantic.size(2), obs_semantic.size(3), obs_semantic.size(4)
            )

        if "semantic_features" in observations:
            x = observations["semantic_features"]
        else:
            # Embed input when using all object categories
            if not self.use_goal_seg:
                categories = observations["semantic"].long() + 1
                observations["semantic"] = self.semantic_embedder(categories)
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)


class ResnetEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=256,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=True,
        spatial_output: bool = False,
        sem_embedding_size=4,
    ):
        super().__init__()
        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            sem_embedding_size=sem_embedding_size,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        obs_rgb = observations["rgb"]
        if len(obs_rgb.size()) == 5:
            observations["rgb"] = obs_rgb.contiguous().view(
                -1, obs_rgb.size(2), obs_rgb.size(3), obs_rgb.size(4)
            )
        obs_depth = observations["depth"]
        if len(obs_rgb.size()) == 5:
            observations["depth"] = obs_depth.contiguous().view(
                -1, obs_depth.size(2), obs_depth.size(3), obs_depth.size(4)
            )
        obs_semantic = observations["semantic"]
        if len(obs_rgb.size()) == 5:
            observations["semantic"] = obs_semantic.contiguous().view(
                -1, obs_semantic.size(2), obs_semantic.size(3), obs_semantic.size(4)
            )

        if "rgb_features" in observations:
            x = observations["rgb_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)
