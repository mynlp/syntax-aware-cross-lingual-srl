import torch
from torch import nn


def swish_activation(inp):
    return inp * torch.sigmoid(inp)


def get_activation(activation):
    if activation == 'tanh':
        return torch.tanh

    if activation == 'relu':
        return nn.ReLU()

    if activation == 'leaky_relu':
        return nn.LeakyReLU()

    if activation == 'elu':
        return nn.ELU()

    if activation == 'swish':
        return swish_activation

    if activation is None:
        return None

    raise Exception(f'Activation function is unknown. '
                    f'Activation function: {activation}.')
