import torch.nn as nn
from models.layer_norm import LayerNorm


class PositionwiseFFNN(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFFNN, self).__init__()
        self.intermediate = nn.Linear(d_model, d_ff)
        self.output = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, inp):
        """
        Layer definition.
        Args:
            inp: [ batch_size, input_len, model_dim ]
        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.relu(self.intermediate(inp))
        output = self.dropout(self.output(inter))
        return self.layer_norm(output + inp)
