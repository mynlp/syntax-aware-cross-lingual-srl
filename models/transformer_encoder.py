import torch.nn as nn
from models.transformer_encoder_layer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    """
    The Transformer encoder from "Attention is All You Need".
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    Returns:
        (`FloatTensor`, `FloatTensor`):
        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_k,
        d_v,
        d_ff,
        dropout,
        max_relative_positions,
        dep_max_relative_positions,
        use_neg_dist,
        use_deprel,
        use_deprel_ext,
        use_dep_path,
        use_dep_ext_path,
        lstm_dropout,
        deprel_size,
        deparc_size,
        deprel_ext_size,
        deprel_edge_dim,
        deparc_edge_dim,
        normalized,
        bias,
        sum_dep_path,
        network_type,
        neighbor_dim,
        omit_self_edge,
        proj_self_edge,
        concat_input,
        apply_gated_head,
        use_positional_embedding,
        use_dep_rel_pos,
        use_word_rel_pos
    ):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        if isinstance(max_relative_positions, int):
            max_relative_positions = [max_relative_positions] * self.num_layers
        assert len(max_relative_positions) == self.num_layers

        if isinstance(dep_max_relative_positions, int):
            dep_max_relative_positions = [dep_max_relative_positions] * self.num_layers
        assert len(dep_max_relative_positions) == self.num_layers

        assert d_k == d_v
        assert d_model == d_k * num_heads

        assert (deprel_edge_dim + deparc_edge_dim) == d_k

        self.layer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    network_type=network_type,
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                    max_relative_position=max_relative_positions[i],
                    dep_max_relative_position=dep_max_relative_positions[i],
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
                    neighbor_dim=neighbor_dim,
                    omit_self_edge=omit_self_edge,
                    proj_self_edge=proj_self_edge,
                    concat_input=concat_input,
                    apply_gated_head=apply_gated_head,
                    use_positional_embedding=use_positional_embedding,
                    use_dep_rel_pos=use_dep_rel_pos,
                    use_word_rel_pos=use_word_rel_pos,
                    lstm_dropout=lstm_dropout
                ) for i in range(num_layers)
            ]
        )

    def count_parameters(self):
        params = list(self.layer.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(
            self,
            inp,
            sent_len_rep,
            adj_mask_rep,
            dep_rel_pos_mat_rep,
            word_rel_pos_mat_rep,
            fc_mask_rep,
            deprel_mat_rep,
            deparc_mat_rep,
            deprel_path_mat_rep,
            deparc_path_mat_rep,
            path_len_mat_rep,
            deprel_ext_mat_rep,
            deprel_ext_path_mat_rep
    ):
        """
        Args:
            inp (`FloatTensor`): `[batch_size x src_len x model_dim]`
            sent_len_rep (`LongTensor`): length of each sequence `[batch]`
            adj_mask_rep (`FloatTensor`): `[batch_size x src_len x src_len]`
            rel_pos_mat_rep (`FloatTensor`): `[batch_size x src_len x src_len]`
            fc_mask_rep (`BoolTensor`): `[batch_size x src_len x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        assert inp.size(0) == sent_len_rep.size(0)

        out = inp
        mask = ~fc_mask_rep
        # Run the forward pass of every layer of the transformers.
        representations = []
        attention_scores = []

        for i in range(self.num_layers):
            out, attn_per_head = self.layer[i](
                inp=out,
                mask=mask,
                adj_mask=adj_mask_rep,
                dep_rel_pos_mat=dep_rel_pos_mat_rep,
                word_rel_pos_mat=word_rel_pos_mat_rep,
                deprel_mat=deprel_mat_rep,
                deparc_mat=deparc_mat_rep,
                deprel_ext_mat=deprel_ext_mat_rep,
                deprel_path_mat=deprel_path_mat_rep,
                deparc_path_mat=deparc_path_mat_rep,
                path_len_mat=path_len_mat_rep,
                deprel_ext_path_mat=deprel_ext_path_mat_rep
            )
            representations.append(out)
            attention_scores.append(attn_per_head)

        return representations, attention_scores
