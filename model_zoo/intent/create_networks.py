from enum import Enum
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.models as models
from torch import nn as nn

try:
    from model_zoo.intent.VGGNet import VGGNet
except ModuleNotFoundError:
    VGGNet = None


class LayerIterator(nn.Module):
    """This class is used to debug NaN error in the image model"""

    def __init__(self, layers):
        super().__init__()
        self.layers: nn.Sequential = layers

    def forward(self, x):
        """Print out info if NaN occurs."""
        last_x = x
        for i, l in enumerate(self.layers):
            name = f"{i}-{str(l)}"
            assert not torch.any(torch.isnan(last_x)), name
            x = l(last_x)
            has_nan = torch.any(torch.isnan(x))
            if has_nan:
                weight_nan = {k: torch.any(t.isnan()) for k, t in l.state_dict().items()}
                max_weight_nan = {k: t.max() for k, t in l.state_dict().items()}
                min_weight_nan = {k: t.min() for k, t in l.state_dict().items()}
                print("last_x.requires_grad", last_x.requires_grad)
                print("last_x", last_x)
                print("last_x.max", last_x.max())
                print("last_x.min", last_x.min())
                print("x.requires_grad", x.requires_grad)
                print("x", x)
                print("x.max", x.max())
                print("x.min", x.min())
                print("weight_nan", weight_nan)
                print("max_weight_nan", max_weight_nan)
                print("min_weight_nan", min_weight_nan)
                print("state_dict", l.state_dict())
            assert not has_nan, name
            last_x = x
        return x


class CNNModel(Enum):
    VGG11 = "vgg11"
    MobileNetV2 = "mobilenetv2"
    CustomNetwork = "custom_network"


def load_cnn(
    model: CNNModel,
    height: int,
    width: int,
    channels: int = 3,
    pretrained: bool = False,
    frozen_layers: int = 0,
    params: Optional[dict] = None,
) -> Tuple[nn.Module, int]:
    """Loads a CNN model (without the fully connected layers at the end).

    Creates a CNN model. The given height and width are is used for ouput size
    computation. VGG11, MobileNetV2, and CustomNetwork are supported options.

    Parameters
    ----------
    model : CNNModel
        The architecture that is created.
    height : int
        Height of the input image for output format computation.
    width : int
        Width of the input image for output format computation.
    channels: int
        The number of color channels to expect as input
    pretrained : bool
        Use pretrained weights for the CNN.
    frozen_layers : int
        Number of frozen layers starting with the first (input) layer.
    params: dict
        A parameters dictionary for custom networks. Currently has layer_features, a list of ints that specifies how many layers and what feature widths do they have.

    Returns
    -------
    nn.Module
        The module containing the CNN backbone.
    int
        Output dimensionality.
    """
    if channels != 3 and model != CNNModel.CustomNetwork:
        raise ValueError(f"Setting channels to anything except 3 is only supported by {CNNModel.CustomNetwork}")

    if model == CNNModel.VGG11:
        if not VGGNet:
            raise ValueError("VGG11 requires VGGNet in the file list")
        back_bone = VGGNet(vgg_pretrained=pretrained, use_classifier=False)
        output_dim = back_bone.update_out_features(height, width)[1]
        if frozen_layers > 0:
            param_dictionary = dict(back_bone.named_parameters())
            param_keys = [
                "model_vgg.features.0.weight",
                "model_vgg.features.0.bias",
                "model_vgg.features.3.weight",
                "model_vgg.features.3.bias",
                "model_vgg.features.6.weight",
                "model_vgg.features.6.bias",
                "model_vgg.features.8.weight",
                "model_vgg.features.8.bias",
                "model_vgg.features.11.weight",
                "model_vgg.features.11.bias",
                "model_vgg.features.13.weight",
                "model_vgg.features.13.bias",
                "model_vgg.features.16.weight",
                "model_vgg.features.16.bias",
                "model_vgg.features.18.weight",
                "model_vgg.features.18.bias",
                "model_vgg.classifier.0.weight",
                "model_vgg.classifier.0.bias",
                "model_vgg.classifier.3.weight",
                "model_vgg.classifier.3.bias",
                "model_vgg.classifier.6.weight",
                "model_vgg.classifier.6.bias",
                "extra_conv.weight",
                "extra_conv.bias",
            ]
            for p_key in param_keys:
                param_dictionary[p_key].requires_grad = True
            for p_key in param_keys[: (frozen_layers + 1)]:
                param_dictionary[p_key].requires_grad = False
    elif model == CNNModel.MobileNetV2:

        assert frozen_layers == 0, "Layer freezing not supported yet for MobileNetv2"

        mobile_net = models.mobilenet_v2(pretrained=pretrained)

        # We do not use the classifier head of mobile net.The main reason for this is
        # that specifying the number of classes and using pretrained weights on all
        # non-classifier layers is not supported by torchvision.
        back_bone = nn.Sequential(mobile_net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(-3, -1))
        output_dim = determine_output_format(back_bone, (1, 3, height, width))[-1]
    elif model == CNNModel.CustomNetwork:
        assert frozen_layers == 0, "Layer freezing not supported yet for CustomNetwork"
        layers = []
        feature_widths = [8, 16, 16, 64, 256]
        if params is not None:
            feature_widths = params["layer_features"]
        layers.append(nn.Conv2d(channels, feature_widths[0], 1, 1))
        w1 = feature_widths[0]
        layers.append(nn.Conv2d(w1, w1, 3, 2, groups=w1))
        for w_i, (w1, w2) in enumerate(zip(feature_widths[:-1], feature_widths[1:])):
            if w_i % 2 == 0:
                groups = 1
            else:
                groups = 1
            # Bottleneck to enforce sparsity.
            w3 = max(int(np.power(2, np.ceil(np.log2(w1) / 2))), 4)
            layers.extend(
                [
                    nn.Conv2d(w1, w3, [1, 3], [1, 1], groups=1),
                    nn.Conv2d(w3, w3, [3, 1], [1, 1], groups=w3),
                    nn.Conv2d(w3, w2, [1, 1], [2, 2], groups=groups),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        layers.append(nn.Conv2d(feature_widths[-1], feature_widths[-1], 1, 1, groups=feature_widths[-1]))
        back_bone = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(-3, -1))
        # Code used to debug NaN in back_bone
        # back_bone = LayerIterator(back_bone)

        output_dim = determine_output_format(back_bone, (1, channels, height, width))[-1]

    else:
        raise ValueError(f"Unsupported Model: {model}")

    return back_bone, output_dim


def determine_output_format(module, input_format):
    """Crude way to determine the output format for a CNN"""
    with torch.no_grad():
        fake_data = torch.ones(input_format)
        return module(fake_data).shape


def create_mlp(
    input_dim: int,
    layers_dim: list,
    dropout_ratio: float,
    batch_norm: bool = False,
    leaky_relu: bool = False,
    pre_bn: bool = False,
    special_init=False,
    special_activation=None,
):
    """Create a Multi Layer Perceptron.

    Parameters
    ----------
    input_dim: int
        Input dimension.
    layers_dim: list
        A list of layer widths.
    dropout_ratio: float
        Dropout ratio.
    batch_norm : bool
        Add batchnorm between modules.
    leaky_relu : bool
        Use leaky ReLU instead of ReLU.
    pre_bn : bool
        Add a batchnorm before the other modules.
    special_init : bool
        If true, use N(0,1) for initializing biases and Glorot initialization for
        weights.

    Returns
    -------
    torch.nn.Sequential
        A sequential container with the resulting MLP.
    """
    output_dim = layers_dim[-1]

    def create_activation():
        if special_activation and special_activation.lower() == "tanh":
            return nn.Tanh()
        if leaky_relu:
            return nn.LeakyReLU(negative_slope=0.1)
        return nn.ReLU()

    if len(layers_dim) == 1:
        res = nn.Linear(input_dim, output_dim)
        res = nn.Sequential(res, create_activation())

    else:
        layers_dim = [input_dim] + layers_dim
        layers = []
        if pre_bn and batch_norm:
            layers.append(nn.BatchNorm1d(layers_dim[0]))
        for l_i, (ld, ld_p1) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            if l_i > 0:
                layers.append(create_activation())
                layers.append(nn.Dropout(p=dropout_ratio))
            new_linear = nn.Linear(ld, ld_p1)
            if special_init:
                nn.init.normal_(new_linear.bias, 0.0, 0.1)
                nn.init.xavier_normal_(new_linear.weight)
            layers.append(new_linear)
            if batch_norm and l_i < (len(layers_dim) - 2):
                layers.append(nn.BatchNorm1d(ld_p1))
        res = nn.Sequential(*layers)
    return res


class MLP(nn.Module):
    def __init__(self, input_size=64, embed_size=64):
        """
        MLP Encoder, including a layer norm layer followed by a relu.
        This is a simplified version of create_mlp.

        Parameters
        ----------
        input_size : int
            Input size.
        embed_size : int
            Embedding size.
        """
        super().__init__()
        self.in_features = input_size
        self.out_features = embed_size
        self.fc_layer = nn.Linear(self.in_features, self.out_features)
        self.layer_norm = nn.LayerNorm(self.out_features)
        self.relu = nn.ReLU()

    def forward(self, input):
        """
        Encode input into an embed vector.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor.

        Returns
        -------
        hidden_states: torch.Tensor
            Embedded tensor.

        """
        hidden_states = self.fc_layer(input)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.relu(hidden_states)
        return hidden_states
