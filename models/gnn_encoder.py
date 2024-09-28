import torch
import torch.nn as nn
from constants import model
from models.edge_dependency_path_encoder import EdgeDependencyPathEncoder
from models.gnn_encoder_layer import GNNEncoderLayer


class GNNEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dropout,
        max_relative_positions,
        use_neg_dist,
        use_deprel,
        use_deprel_ext,
        use_dep_path,
        use_dep_ext_path,
        lstm_dropout,
        deprel_size,
        deprel_ext_size,
        deparc_size,
        deprel_edge_dim,
        deparc_edge_dim,
        normalized,
        bias,
        init_weight_xavier_uniform,
        concat_at_final_layer,
        activation_at_final_layer,
        activation,
        network_type,
        dropout_final,
        residual,
        layer_norm,
        final_bias,
        deparc_voc,
        deprel_ext_voc,
        base_size,
        use_positional_embedding,
        deprel_ext_edge_dim,
        rel_pos_dim,
        att_dim,
        sum_dep_path,
        omit_self_edge,
        proj_self_edge,
        d_k,
        d_v,
        neighbor_dim,
        concat_input,
        apply_gated_head,
        same_relation_dim,
        use_dep_rel_pos,
        use_word_rel_pos,
        average_heads
    ):
        super(GNNEncoder, self).__init__()

        self.num_layers = num_layers
        if isinstance(max_relative_positions, int):
            max_relative_positions = [max_relative_positions] * self.num_layers
        assert len(max_relative_positions) == self.num_layers

        self.use_dep_path = use_dep_path
        self.use_dep_ext_path = use_dep_ext_path
        self.use_deprel = use_deprel
        self.use_deprel_ext = use_deprel_ext
        self.use_dep_rel_pos = use_dep_rel_pos
        self.use_word_rel_pos = use_word_rel_pos

        if activation == 'elu':
            activation_function = nn.ELU()
        elif activation == 'leaky_relu':
            activation_function = nn.LeakyReLU()
        else:
            activation_function = nn.ReLU()

        layers = []

        for i in range(num_layers):
            if average_heads:
                concat = False
            else:
                concat = (i < num_layers - 1 or concat_at_final_layer)

            layers.append(GNNEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                max_relative_position=max_relative_positions[i],
                use_neg_dist=use_neg_dist,
                use_deprel=use_deprel,
                use_deprel_ext=use_deprel_ext,
                use_dep_path=use_dep_path,
                use_dep_ext_path=use_dep_ext_path,
                deprel_size=deprel_size,
                deprel_ext_size=deprel_ext_size,
                deparc_size=deparc_size,
                deprel_edge_dim=deprel_edge_dim,
                deparc_edge_dim=deparc_edge_dim,
                normalized=normalized,
                bias=bias,
                init_weight_xavier_uniform=init_weight_xavier_uniform,
                concat=concat,
                activation=activation_function if (i < num_layers - 1 or activation_at_final_layer) else None,
                network_type=network_type,
                dropout_final=dropout_final,
                residual=residual,
                layer_norm=layer_norm,
                final_bias=final_bias,
                deparc_voc=deparc_voc,
                deprel_ext_voc=deprel_ext_voc,
                base_size=base_size,
                use_positional_embedding=use_positional_embedding,
                deprel_ext_edge_dim=deprel_ext_edge_dim,
                rel_pos_dim=rel_pos_dim,
                att_dim=att_dim,
                layer_num=i,
                omit_self_edge=omit_self_edge,
                proj_self_edge=proj_self_edge,
                neighbor_dim=neighbor_dim,
                d_k=d_k,
                d_v=d_v,
                concat_input=concat_input,
                apply_gated_head=apply_gated_head,
                same_relation_dim=same_relation_dim,
                use_dep_rel_pos=use_dep_rel_pos,
                use_word_rel_pos=use_word_rel_pos,
                lstm_dropout=lstm_dropout,
                sum_dep_path=sum_dep_path
            ))

        self.layer = nn.ModuleList(layers)

    def count_parameters(self):
        params = list(self.layer.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(
            self,
            inp,
            sent_len_rep,
            edge_len_rep,
            edge_index_rep,
            dep_rel_pos_edge_rep,
            word_rel_pos_edge_rep,
            deprel_edge_rep,
            deparc_edge_rep,
            deprel_path_edge_rep,
            deparc_path_edge_rep,
            path_len_edge_rep,
            dist_edge_rep,
            deprel_ext_edge_rep,
            deprel_ext_path_edge_rep
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

        batch_size, max_sent_len, model_dim = inp.size()
        out = inp.view(batch_size * max_sent_len, model_dim)

        # Run the forward pass of every layer of the transformers.
        representations = []
        attention_scores = []

        edge_len_total = torch.sum(edge_len_rep).item()

        reshaped_dep_rel_pos_edge_rep = None
        reshaped_word_rel_pos_edge_rep = None
        reshaped_dist_edge_rep = torch.zeros(edge_len_total).to(dist_edge_rep)
        if self.use_dep_rel_pos:
            reshaped_dep_rel_pos_edge_rep = torch.zeros(edge_len_total).to(dep_rel_pos_edge_rep)
        if self.use_word_rel_pos:
            reshaped_word_rel_pos_edge_rep = torch.zeros(edge_len_total).to(word_rel_pos_edge_rep)
        reshaped_deparc_edge_rep = torch.zeros(edge_len_total).to(deparc_edge_rep)
        reshaped_edge_index_rep = torch.zeros((edge_len_total, 2)).to(edge_index_rep)
        reshaped_deprel_edge_rep = torch.zeros(edge_len_total).to(deprel_edge_rep) if self.use_deprel else None
        reshaped_deprel_ext_edge_rep = torch.zeros(edge_len_total).to(deprel_ext_edge_rep) if self.use_deprel_ext \
            else None
        reshaped_deprel_path_edge_rep = torch.zeros((edge_len_total, deprel_path_edge_rep.size(-1))).to(deprel_path_edge_rep) if self.use_dep_path \
            else None
        reshaped_deparc_path_edge_rep = torch.zeros((edge_len_total, deparc_path_edge_rep.size(-1))).to(deparc_path_edge_rep) if self.use_dep_path \
            else None
        reshaped_deprel_ext_path_edge_rep = torch.zeros((edge_len_total, deprel_ext_path_edge_rep.size(-1))).to(deprel_ext_path_edge_rep) if self.use_dep_ext_path \
            else None
        reshaped_path_len_edge_rep = torch.zeros(edge_len_total).to(path_len_edge_rep) if self.use_dep_path or self.use_dep_ext_path \
            else None

        start_idx = 0

        for idx in range(len(edge_len_rep)):
            actual_len = edge_len_rep[idx]
            if actual_len == 0:
                continue

            edge_idx_offset = idx * max_sent_len

            end_idx = start_idx + actual_len

            reshaped_edge_index_rep[start_idx:end_idx] = edge_index_rep[idx][:actual_len] + edge_idx_offset

            if self.use_dep_rel_pos:
                reshaped_dep_rel_pos_edge_rep[start_idx:end_idx] = dep_rel_pos_edge_rep[idx][:actual_len]

            if self.use_word_rel_pos:
                reshaped_word_rel_pos_edge_rep[start_idx:end_idx] = word_rel_pos_edge_rep[idx][:actual_len]

            reshaped_dist_edge_rep[start_idx:end_idx] = dist_edge_rep[idx][:actual_len]
            reshaped_deparc_edge_rep[start_idx:end_idx] = deparc_edge_rep[idx][:actual_len]

            if self.use_deprel:
                reshaped_deprel_edge_rep[start_idx:end_idx] = deprel_edge_rep[idx][:actual_len]

            if self.use_deprel_ext:
                reshaped_deprel_ext_edge_rep[start_idx:end_idx] = deprel_ext_edge_rep[idx][:actual_len]

            if self.use_dep_path or self.use_dep_ext_path:
                reshaped_path_len_edge_rep[start_idx:end_idx] = path_len_edge_rep[idx][:actual_len]

            if self.use_dep_path:
                reshaped_deprel_path_edge_rep[start_idx:end_idx] = deprel_path_edge_rep[idx][:actual_len]
                reshaped_deparc_path_edge_rep[start_idx:end_idx] = deparc_path_edge_rep[idx][:actual_len]

            if self.use_dep_ext_path:
                reshaped_deprel_ext_path_edge_rep[start_idx:end_idx] = deprel_ext_path_edge_rep[idx][:actual_len]

            start_idx = end_idx

        assert start_idx == edge_len_total
        reshaped_relations = None

        for i in range(self.num_layers):
            out, attn_per_head, next_relations = self.layer[i](
                inp=out,
                dep_rel_pos_edge=reshaped_dep_rel_pos_edge_rep,
                word_rel_pos_edge=reshaped_word_rel_pos_edge_rep,
                deprel_edge=reshaped_deprel_edge_rep,
                deprel_ext_edge=reshaped_deprel_ext_edge_rep,
                deparc_edge=reshaped_deparc_edge_rep,
                edge_index=reshaped_edge_index_rep.transpose(0, 1),
                relations=reshaped_relations,
                dist_edge=reshaped_dist_edge_rep,
                deprel_path_edge=reshaped_deprel_path_edge_rep,
                deparc_path_edge=reshaped_deparc_path_edge_rep,
                path_len_edge=reshaped_path_len_edge_rep,
                deprel_ext_path_edge=reshaped_deprel_ext_path_edge_rep
            )

            if next_relations is not None:
                reshaped_relations = next_relations

            representations.append(out.view(batch_size, max_sent_len, -1))
            attention_scores.append(attn_per_head)

        return representations, attention_scores
