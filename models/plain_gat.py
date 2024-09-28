import torch
import torch.nn as nn
import torch.nn.functional as f
from utils import model
from constants import word


class PlainGAT(nn.Module):
    def __init__(
        self,
        num_heads,
        model_dim,
        dropout,
        normalized,
        concat,
        activation,
        dropout_init,
        dropout_final,
        final_bias,
        bias,
        init_weight_xavier_uniform
    ):
        super(PlainGAT, self).__init__()

        self.is_dropout_init = dropout_init
        self.is_dropout_final = dropout_final
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.is_concat = concat
        self.is_final_bias = final_bias
        self.activation = activation
        self.is_init_weight_xavier_uniform = init_weight_xavier_uniform

        assert self.model_dim % self.num_heads == 0
        if self.is_concat:
            self.d_v = int(self.model_dim / self.num_heads)
        else:
            self.d_v = self.model_dim

        if self.is_final_bias:
            self.final_bias = nn.Parameter(torch.Tensor(self.model_dim))
        else:
            self.final_bias = None

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.value = nn.Linear(model_dim, num_heads * self.d_v, bias=bias)
        self.weight_vector_value_src = nn.Parameter(torch.Tensor(1, 1, 1, self.num_heads, self.d_v))
        self.weight_vector_value_tgt = nn.Parameter(torch.Tensor(1, 1, 1, self.num_heads, self.d_v))

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.normalized = normalized

        self.init_params()

    def init_params(self):
        if self.is_init_weight_xavier_uniform:
            nn.init.xavier_uniform_(self.value.weight)

        nn.init.xavier_uniform_(self.weight_vector_value_src)
        nn.init.xavier_uniform_(self.weight_vector_value_tgt)

        if self.is_final_bias:
            torch.nn.init.zeros_(self.final_bias)

    def forward(
        self,
        inp,
        mask,
        adj_mask
    ):
        batch_size = inp.size(0)
        num_heads = self.num_heads

        if self.is_dropout_init:
            do_inp = self.dropout(inp)
        else:
            do_inp = inp

        # 1) Project value.
        value = model.shape(
            x=self.value(do_inp),
            dim=self.d_v,
            batch_size=batch_size,
            num_heads=num_heads
        )  # bsz x nhead x value_len x d_v

        if self.is_dropout_init:
            value = self.dropout(value)

        # 2) Concatenate source and target node representations, multiply with weight vector, and apply leaky ReLU.
        _, _, value_len, _ = value.size()  # bsz x nhead x value_len x d_v
        # bsz x nhead x value_len x value_len x d_v
        value_src = value.unsqueeze(2).repeat(1, 1, value_len, 1, 1)
        # bsz x nhead x value_len x value_len x d_v
        value_tgt = value.unsqueeze(3).repeat(1, 1, 1, value_len, 1)

        # bsz x value_len x value_len x nhead x d_v
        value_src = value_src.permute(0, 2, 3, 1, 4)
        # bsz x value_len x value_len x nhead x d_v
        value_tgt = value_tgt.permute(0, 2, 3, 1, 4)

        # bsz x value_len x value_len x nhead
        scores_src = (value_src * self.weight_vector_value_src).sum(dim=-1)
        # bsz x value_len x value_len x nhead
        scores_tgt = (value_tgt * self.weight_vector_value_tgt).sum(dim=-1)
        # bsz x nhead x value_len x value_len
        scores_value = (scores_src + scores_tgt).permute(0, 3, 1, 2)
        scores_value = self.leakyReLU(scores_value)

        # 3) Apply mask for padding.
        mask = mask.unsqueeze(1)  # [B, 1, query_len, key_len]
        scores_value = scores_value.masked_fill(mask, -word.INFINITY_NUMBER)

        if adj_mask is not None:
            assert adj_mask.size() == scores_value.size()
            scores_value = scores_value.masked_fill(~adj_mask.bool(), -word.INFINITY_NUMBER)

        # 4) Apply softmax over neighborhood and normalize.
        attn_value = self.softmax(scores_value).to(value.dtype)

        if adj_mask is not None and self.normalized:
            assert adj_mask.size() == attn_value.size()

            adj_mask_ = adj_mask.masked_fill(~adj_mask.bool(), word.INFINITY_NUMBER)
            adj_mask_ = 1.0 / adj_mask_

            attn_value = attn_value * adj_mask_
            attn_value = f.normalize(attn_value, p=1, dim=-1)

        # TODO: Check if this is true.
        # To connect only desirable edges.
        attn_value = attn_value.masked_fill(mask, 0)
        if adj_mask is not None:
            attn_value = attn_value.masked_fill(~adj_mask.bool(), 0)

        # bsz x nhead x value_len x value_len
        attn_value = self.dropout(attn_value)

        # bsz x nhead x value_len x d_v
        output_original_value = torch.matmul(attn_value, value)

        if self.is_concat:
            # bsz x value_len x model_dim (=num_heads x d_v)
            output_value = model.unshape(
                x=output_original_value,
                dim=self.d_v,
                batch_size=batch_size,
                num_heads=num_heads
            )
        else:
            # bsz x value_len x model_dim (=d_v)
            output_value = output_original_value.mean(dim=1)

        if self.is_dropout_final:
            output_value = self.dropout(output_value)

        # Skip connection
        final_output = output_value + inp

        if self.is_final_bias:
            final_output = final_output + self.final_bias

        if self.activation is not None:
            final_output = self.activation(final_output)

        # a list of size num_heads containing tensors
        # of shape `batch x query_len x key_len`
        attn_value_per_head = [attn.squeeze(1) for attn in attn_value.chunk(num_heads, dim=1)]

        return final_output, attn_value_per_head
