import torch
import torch.nn as nn
from models.layer_norm import LayerNorm


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        bidirectional,
        activation,
        dropout_final,
        residual,
        layer_norm
    ):
        super(LSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.out_size = hidden_size * 2 if bidirectional else hidden_size

        self.activation = activation
        self.is_dropout_final = dropout_final
        self.is_layer_norm = layer_norm

        self.layer_norm = LayerNorm(self.out_size) if self.is_layer_norm else None

        self.dropout = nn.Dropout(dropout)
        self.is_residual = residual

    def count_parameters(self):
        # TODO: Implement count parameters
        pass

    def forward(
        self,
        inp
    ):
        output, _ = self.lstm(inp)

        if self.activation is not None:
            output = self.activation(output)

        if self.is_dropout_final:
            output = self.dropout(output)

        if self.is_residual:
            if output.shape[-1] > inp.shape[-1]:
                output = output + torch.cat([inp, inp], dim=-1)
            else:
                output = output + inp

        if self.is_layer_norm:
            output = self.layer_norm(output)

        return output.contiguous()
