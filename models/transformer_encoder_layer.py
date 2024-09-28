import torch.nn as nn
from models.multi_head_attention import MultiHeadedAttention
from models.multi_head_attention_gaan import MultiHeadedAttentionGaan
from models.layer_norm import LayerNorm
from models.positionwise_ffnn import PositionwiseFFNN
from constants import model


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(
            self,
            network_type,
            d_model,
            num_heads,
            d_ff,
            d_k,
            d_v,
            dropout,
            max_relative_position,
            dep_max_relative_position,
            use_neg_dist,
            use_deprel,
            use_dep_path,
            use_deprel_ext,
            use_dep_ext_path,
            sum_dep_path,
            deprel_size,
            deparc_size,
            deprel_ext_size,
            deprel_edge_dim,
            deparc_edge_dim,
            normalized,
            bias,
            neighbor_dim,
            omit_self_edge,
            proj_self_edge,
            concat_input,
            apply_gated_head,
            use_positional_embedding,
            use_dep_rel_pos,
            use_word_rel_pos,
            lstm_dropout
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.network_type = network_type

        if self.network_type == model.network_type['trans']:
            self.trans = MultiHeadedAttention(
                num_heads=num_heads,
                model_dim=d_model,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout,
                max_relative_position=max_relative_position,
                dep_max_relative_position=dep_max_relative_position,
                use_neg_dist=use_neg_dist,
                use_deprel=use_deprel,
                use_dep_path=use_dep_path,
                use_deprel_ext=use_deprel_ext,
                use_dep_ext_path=use_dep_ext_path,
                sum_dep_path=sum_dep_path,
                deprel_size=deprel_size,
                deparc_size=deparc_size,
                deprel_ext_size=deprel_ext_size,
                deprel_edge_dim=deprel_edge_dim,
                deparc_edge_dim=deparc_edge_dim,
                normalized=normalized,
                bias=bias,
                omit_self_edge=omit_self_edge,
                proj_self_edge=proj_self_edge,
                concat_input=concat_input,
                neighbor_dim=neighbor_dim,
                apply_gated_head=apply_gated_head,
                use_positional_embedding=use_positional_embedding,
                use_dep_rel_pos=use_dep_rel_pos,
                use_word_rel_pos=use_word_rel_pos,
                lstm_dropout=lstm_dropout
            )
        elif self.network_type == model.network_type['trans_gaan']:
            self.trans_gaan = MultiHeadedAttentionGaan(
                num_heads=num_heads,
                model_dim=d_model,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout,
                max_relative_position=max_relative_position,
                dep_max_relative_position=dep_max_relative_position,
                use_neg_dist=use_neg_dist,
                use_deprel=use_deprel,
                use_dep_path=use_dep_path,
                use_deprel_ext=use_deprel_ext,
                use_dep_ext_path=use_dep_ext_path,
                sum_dep_path=sum_dep_path,
                deprel_size=deprel_size,
                deparc_size=deparc_size,
                deprel_ext_size=deprel_ext_size,
                deprel_edge_dim=deprel_edge_dim,
                deparc_edge_dim=deparc_edge_dim,
                neighbor_dim=neighbor_dim,
                normalized=normalized,
                bias=bias,
                omit_self_edge=omit_self_edge,
                proj_self_edge=proj_self_edge,
                concat_input=concat_input,
                use_positional_embedding=use_positional_embedding,
                use_dep_rel_pos=use_dep_rel_pos,
                use_word_rel_pos=use_word_rel_pos,
                lstm_dropout=lstm_dropout
            )
        else:
            raise Exception(f'Network type is unknown. Network type: {self.network_type}.')

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        self.feed_forward = PositionwiseFFNN(d_model, d_ff, dropout)

    def forward(
            self,
            inp,
            mask,
            adj_mask,
            dep_rel_pos_mat,
            word_rel_pos_mat,
            deprel_mat,
            deparc_mat,
            deprel_ext_mat,
            deprel_path_mat,
            deparc_path_mat,
            path_len_mat,
            deprel_ext_path_mat
    ):
        """
        Transformer Encoder Layer definition.
        Args:
            inp (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`BoolTensor`): `[batch_size x src_len x src_len]`
            adj_mask (`FloatTensor`): `[batch_size x src_len x src_len]`
            rel_pos_mat (`FloatTensor`): `[batch_size x src_len x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        if self.network_type == model.network_type['trans']:
            context, attn_per_head = self.trans(
                key=inp,
                value=inp,
                query=inp,
                mask=mask,
                adj_mask=adj_mask,
                dep_rel_pos_mat=dep_rel_pos_mat,
                word_rel_pos_mat=word_rel_pos_mat,
                deprel_mat=deprel_mat,
                deparc_mat=deparc_mat,
                deprel_ext_mat=deprel_ext_mat,
                deprel_path_mat=deprel_path_mat,
                deparc_path_mat=deparc_path_mat,
                path_len_mat=path_len_mat,
                deprel_ext_path_mat=deprel_ext_path_mat
            )
        elif self.network_type == model.network_type['trans_gaan']:
            context, attn_per_head = self.trans_gaan(
                key=inp,
                value=inp,
                query=inp,
                mask=mask,
                adj_mask=adj_mask,
                dep_rel_pos_mat=dep_rel_pos_mat,
                word_rel_pos_mat=word_rel_pos_mat,
                deprel_mat=deprel_mat,
                deparc_mat=deparc_mat,
                deprel_ext_mat=deprel_ext_mat,
                deprel_path_mat=deprel_path_mat,
                deparc_path_mat=deparc_path_mat,
                path_len_mat=path_len_mat,
                deprel_ext_path_mat=deprel_ext_path_mat
            )
        else:
            raise Exception(f'Network type is unknown. Network type: {self.network_type}.')

        out = self.layer_norm(self.dropout(context) + inp)

        return self.feed_forward(out), attn_per_head
