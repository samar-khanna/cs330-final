import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_model(num_conv_layers, num_in_channels, num_hidden_channels, num_outputs,
               kernel_size, device):
    """
    The network consists of four convolutional blocks followed by a linear
    head layer. Each convolutional block comprises a convolution layer, a
    batch normalization layer, and ReLU activation.

    Note that unlike conventional use, batch normalization is always done
    with batch statistics, regardless of whether we are training or
    evaluating. This technically makes meta-learning transductive, as
    opposed to inductive.
    :param num_conv_layers:
    :param num_in_channels:
    :param num_hidden_channels:
    :param num_outputs:
    :param kernel_size:
    :param device:
    :return:
    """
    meta_parameters = {}

    # construct feature extractor
    in_channels = num_in_channels
    for i in range(num_conv_layers):
        meta_parameters[f'conv{i}'] = nn.init.xavier_uniform_(
            torch.empty(
                num_hidden_channels,
                in_channels,
                kernel_size,
                kernel_size,
                requires_grad=True,
                device=device
            )
        )
        meta_parameters[f'b{i}'] = nn.init.zeros_(
            torch.empty(
                num_hidden_channels,
                requires_grad=True,
                device=device
            )
        )
        in_channels = num_hidden_channels

    # construct linear head layer
    meta_parameters[f'w{num_conv_layers}'] = nn.init.xavier_uniform_(
        torch.empty(
            num_outputs,
            num_hidden_channels,
            requires_grad=True,
            device=device
        )
    )
    meta_parameters[f'b{num_conv_layers}'] = nn.init.zeros_(
        torch.empty(
            num_outputs,
            requires_grad=True,
            device=device
        )
    )

    def forward(x, params):
        for i in range(num_conv_layers):
            x = F.conv2d(
                input=x,
                weight=params[f'conv{i}'],
                bias=params[f'b{i}'],
                stride=1,
                padding='same'
            )
            x = F.batch_norm(x, None, None, training=True)
            x = F.relu(x)
        x = torch.mean(x, dim=[2, 3])
        return F.linear(
            input=x,
            weight=params[f'w{num_conv_layers}'],
            bias=params[f'b{num_conv_layers}']
        )

    return meta_parameters, forward
