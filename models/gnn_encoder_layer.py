import torch.nn as nn
from constants import model
from models.plain_gat_2 import PlainGAT
from models.heterogeneous_gat_2 import HeterogeneousGAT
from models.two_attention_gat_2 import TwoAttentionGAT
from models.two_attention_gat_original import TwoAttentionGATOriginal
from models.syntactic_gcn import SyntacticGCN
from models.relational_gcn import RelationalGCN
from models.argcn import ARGCN
from models.kbgat import KBGAT
from models.gaan import GaAN
from models.san import SAN
from models.layer_norm import LayerNorm


class GNNEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dropout,
        max_relative_position,
        use_neg_dist,
        use_deprel,
        use_deprel_ext,
        use_dep_path,
        use_dep_ext_path,
        deprel_size,
        deprel_ext_size,
        deparc_size,
        deprel_edge_dim,
        deparc_edge_dim,
        normalized,
        bias,
        init_weight_xavier_uniform,
        concat,
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
        layer_num,
        omit_self_edge,
        proj_self_edge,
        neighbor_dim,
        d_k,
        d_v,
        concat_input,
        apply_gated_head,
        same_relation_dim,
        use_dep_rel_pos,
        use_word_rel_pos,
        lstm_dropout,
        sum_dep_path
    ):
        super(GNNEncoderLayer, self).__init__()
        self.network_type = network_type
        self.activation = activation
        self.is_dropout_final = dropout_final
        self.is_residual = residual
        self.is_layer_norm = layer_norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model) if self.is_layer_norm else None

        if self.network_type == model.gat_type['gat_plain']:
            self.gat_plain = PlainGAT(
                deparc_voc=deparc_voc,
                num_heads=num_heads,
                model_dim=d_model,
                dropout=dropout,
                normalized=normalized,
                concat=concat,
                final_bias=final_bias,
                omit_self_edge=omit_self_edge,
                proj_self_edge=proj_self_edge,
                concat_input=concat_input,
                apply_gated_head=apply_gated_head,
                neighbor_dim=neighbor_dim,
                bias=bias
            )
        elif self.network_type == model.gat_type['gat_het']:
            self.gat_het = HeterogeneousGAT(
                deparc_voc=deparc_voc,
                deprel_ext_size=deprel_ext_size,
                num_heads=num_heads,
                model_dim=d_model,
                dropout=dropout,
                max_relative_position=max_relative_position,
                use_neg_dist=use_neg_dist,
                use_deprel=use_deprel,
                use_deprel_ext=use_deprel_ext,
                deprel_size=deprel_size,
                deparc_size=deparc_size,
                deprel_edge_dim=deprel_edge_dim,
                deparc_edge_dim=deparc_edge_dim,
                normalized=normalized,
                concat=concat,
                final_bias=final_bias,
                rel_pos_dim=rel_pos_dim,
                use_dep_path=use_dep_path,
                use_dep_ext_path=use_dep_ext_path,
                omit_self_edge=omit_self_edge,
                proj_self_edge=proj_self_edge,
                concat_input=concat_input,
                apply_gated_head=apply_gated_head,
                neighbor_dim=neighbor_dim,
                bias=bias,
                use_positional_embedding=use_positional_embedding,
                same_relation_dim=same_relation_dim,
                deprel_ext_edge_dim=deprel_ext_edge_dim,
                use_dep_rel_pos=use_dep_rel_pos,
                use_word_rel_pos=use_word_rel_pos,
                lstm_dropout=lstm_dropout,
                sum_dep_path=sum_dep_path
            )
        elif self.network_type == model.gat_type['gat_two_att']:
            self.gat_two_att = TwoAttentionGAT(
                deparc_voc=deparc_voc,
                deprel_ext_size=deprel_ext_size,
                num_heads=num_heads,
                model_dim=d_model,
                dropout=dropout,
                max_relative_position=max_relative_position,
                use_neg_dist=use_neg_dist,
                use_deprel=use_deprel,
                use_deprel_ext=use_deprel_ext,
                deprel_size=deprel_size,
                deparc_size=deparc_size,
                deprel_edge_dim=deprel_edge_dim,
                deparc_edge_dim=deparc_edge_dim,
                normalized=normalized,
                concat=concat,
                rel_pos_dim=rel_pos_dim,
                use_dep_path=use_dep_path,
                use_dep_ext_path=use_dep_ext_path,
                omit_self_edge=omit_self_edge,
                proj_self_edge=proj_self_edge,
                concat_input=concat_input,
                apply_gated_head=apply_gated_head,
                neighbor_dim=neighbor_dim,
                bias=bias,
                use_positional_embedding=use_positional_embedding,
                same_relation_dim=same_relation_dim,
                final_bias=final_bias,
                deprel_ext_edge_dim=deprel_ext_edge_dim,
                use_dep_rel_pos=use_dep_rel_pos,
                use_word_rel_pos=use_word_rel_pos,
                lstm_dropout=lstm_dropout,
                sum_dep_path=sum_dep_path
            )
        elif self.network_type == model.gat_type['gat_two_att_ori']:
            self.gat_two_att_ori = TwoAttentionGATOriginal(
                deparc_voc=deparc_voc,
                deprel_ext_size=deprel_ext_size,
                num_heads=num_heads,
                model_dim=d_model,
                dropout=dropout,
                max_relative_position=max_relative_position,
                use_neg_dist=use_neg_dist,
                use_deprel=use_deprel,
                use_deprel_ext=use_deprel_ext,
                deprel_size=deprel_size,
                deparc_size=deparc_size,
                deprel_edge_dim=deprel_edge_dim,
                deparc_edge_dim=deparc_edge_dim,
                normalized=normalized,
                concat=concat,
                rel_pos_dim=rel_pos_dim,
                use_dep_path=use_dep_path,
                use_dep_ext_path=use_dep_ext_path,
                omit_self_edge=omit_self_edge,
                proj_self_edge=proj_self_edge,
                concat_input=concat_input,
                apply_gated_head=apply_gated_head,
                neighbor_dim=neighbor_dim,
                bias=bias,
                use_positional_embedding=use_positional_embedding,
                same_relation_dim=same_relation_dim,
                final_bias=final_bias,
                deprel_ext_edge_dim=deprel_ext_edge_dim,
                use_dep_rel_pos=use_dep_rel_pos,
                use_word_rel_pos=use_word_rel_pos,
                lstm_dropout=lstm_dropout,
                sum_dep_path=sum_dep_path
            )
        elif self.network_type == model.gat_type['gat_kb']:
            self.gat_kb = KBGAT(
                deprel_ext_size=deprel_ext_size,
                deparc_voc=deparc_voc,
                deprel_edge_dim=deprel_edge_dim,
                deparc_edge_dim=deparc_edge_dim,
                max_relative_position=max_relative_position,
                use_neg_dist=use_neg_dist,
                use_deprel=use_deprel,
                use_deprel_ext=use_deprel_ext,
                deprel_size=deprel_size,
                deparc_size=deparc_size,
                model_dim=d_model,
                num_heads=num_heads,
                final_bias=final_bias,
                dropout=dropout,
                concat=concat,
                normalized=normalized,
                init_weight_xavier_uniform=init_weight_xavier_uniform,
                layer_num=layer_num,
                rel_pos_dim=rel_pos_dim,
                use_dep_path=use_dep_path,
                use_dep_ext_path=use_dep_ext_path,
                omit_self_edge=omit_self_edge,
                proj_self_edge=proj_self_edge,
                concat_input=concat_input,
                apply_gated_head=apply_gated_head,
                neighbor_dim=neighbor_dim,
                bias=bias,
                use_positional_embedding=use_positional_embedding,
                deprel_ext_edge_dim=deprel_ext_edge_dim,
                use_dep_rel_pos=use_dep_rel_pos,
                use_word_rel_pos=use_word_rel_pos,
                lstm_dropout=lstm_dropout,
                sum_dep_path=sum_dep_path
            )
        elif self.network_type == model.gat_type['gaan']:
            self.gaan = GaAN(
                deparc_voc=deparc_voc,
                num_heads=num_heads,
                model_dim=d_model,
                neighbor_dim=neighbor_dim,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout,
                normalized=normalized,
                bias=bias,
                max_relative_position=max_relative_position,
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
                omit_self_edge=omit_self_edge,
                proj_self_edge=proj_self_edge,
                concat_input=concat_input,
                use_positional_embedding=use_positional_embedding,
                use_dep_rel_pos=use_dep_rel_pos,
                use_word_rel_pos=use_word_rel_pos,
                lstm_dropout=lstm_dropout,
                sum_dep_path=sum_dep_path
            )
        elif self.network_type == model.gat_type['san']:
            self.san = SAN(
                deparc_voc=deparc_voc,
                num_heads=num_heads,
                model_dim=d_model,
                neighbor_dim=neighbor_dim,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout,
                normalized=normalized,
                bias=bias,
                max_relative_position=max_relative_position,
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
                omit_self_edge=omit_self_edge,
                proj_self_edge=proj_self_edge,
                concat_input=concat_input,
                apply_gated_head=apply_gated_head,
                use_positional_embedding=use_positional_embedding,
                use_dep_rel_pos=use_dep_rel_pos,
                use_word_rel_pos=use_word_rel_pos,
                lstm_dropout=lstm_dropout,
                sum_dep_path=sum_dep_path
            )
        elif self.network_type == model.gcn_type['gcn_syntactic']:
            self.gcn_syntactic = SyntacticGCN(
                deparc_voc=deparc_voc,
                deprel_size=deprel_size,
                model_dim=d_model,
                use_deprel=use_deprel,
                init_weight_xavier_uniform=init_weight_xavier_uniform
            )
        elif self.network_type == model.gcn_type['gcn_relational']:
            self.gcn_relational = RelationalGCN(
                deprel_ext_size=deprel_ext_size,
                deprel_ext_voc=deprel_ext_voc,
                base_size=base_size,
                model_dim=d_model,
                final_bias=final_bias,
                use_deprel_ext=use_deprel_ext,
                omit_self_edge=omit_self_edge,
                proj_self_edge=proj_self_edge
            )
        elif self.network_type == model.gcn_type['gcn_ar']:
            self.gcn_ar = ARGCN(
                deprel_ext_size=deprel_ext_size,
                deprel_ext_voc=deprel_ext_voc,
                use_neg_dist=use_neg_dist,
                max_relative_position=max_relative_position,
                use_positional_embedding=use_positional_embedding,
                deprel_ext_edge_dim=deprel_ext_edge_dim,
                use_deprel=use_deprel,
                deprel_size=deprel_size,
                deparc_size=deparc_size,
                deprel_edge_dim=deprel_edge_dim,
                deparc_edge_dim=deparc_edge_dim,
                use_deprel_ext=use_deprel_ext,
                rel_pos_dim=rel_pos_dim,
                att_dim=att_dim,
                model_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                final_bias=final_bias,
                omit_self_edge=omit_self_edge,
                proj_self_edge=proj_self_edge,
                concat_input=concat_input,
                apply_gated_head=apply_gated_head,
                neighbor_dim=neighbor_dim,
                bias=bias,
                d_v=d_v,
                use_dep_rel_pos=use_dep_rel_pos,
                use_word_rel_pos=use_word_rel_pos
            )
        else:
            raise Exception(f'GNN type is unknown. GNN type: {self.network_type}.')

    def forward(
            self,
            inp,
            dep_rel_pos_edge,
            word_rel_pos_edge,
            deprel_edge,
            deprel_ext_edge,
            deparc_edge,
            edge_index,
            relations,
            dist_edge,
            deprel_path_edge,
            deparc_path_edge,
            path_len_edge,
            deprel_ext_path_edge
    ):
        attn_per_head = None
        next_relations = None
        if self.network_type == model.gat_type['gat_plain']:
            context, attn_per_head = self.gat_plain(
                inp=inp,
                edge_index=edge_index,
                dist_edge=dist_edge,
                deparc_edge=deparc_edge
            )
        elif self.network_type == model.gat_type['gat_het']:
            context, attn_per_head = self.gat_het(
                inp=inp,
                dep_rel_pos_edge=dep_rel_pos_edge,
                word_rel_pos_edge=word_rel_pos_edge,
                deprel_edge=deprel_edge,
                deprel_ext_edge=deprel_ext_edge,
                deparc_edge=deparc_edge,
                edge_index=edge_index,
                dist_edge=dist_edge,
                deprel_path_edge=deprel_path_edge,
                deparc_path_edge=deparc_path_edge,
                path_len_edge=path_len_edge,
                deprel_ext_path_edge=deprel_ext_path_edge
            )
        elif self.network_type == model.gat_type['gat_two_att']:
            context, attn_per_head, _ = self.gat_two_att(
                inp=inp,
                dep_rel_pos_edge=dep_rel_pos_edge,
                word_rel_pos_edge=word_rel_pos_edge,
                deprel_edge=deprel_edge,
                deprel_ext_edge=deprel_ext_edge,
                deparc_edge=deparc_edge,
                edge_index=edge_index,
                dist_edge=dist_edge,
                deprel_path_edge=deprel_path_edge,
                deparc_path_edge=deparc_path_edge,
                path_len_edge=path_len_edge,
                deprel_ext_path_edge=deprel_ext_path_edge
            )
        elif self.network_type == model.gat_type['gat_two_att_ori']:
            context, attn_per_head, _ = self.gat_two_att_ori(
                inp=inp,
                dep_rel_pos_edge=dep_rel_pos_edge,
                word_rel_pos_edge=word_rel_pos_edge,
                deprel_edge=deprel_edge,
                deprel_ext_edge=deprel_ext_edge,
                deparc_edge=deparc_edge,
                edge_index=edge_index,
                dist_edge=dist_edge,
                deprel_path_edge=deprel_path_edge,
                deparc_path_edge=deparc_path_edge,
                path_len_edge=path_len_edge,
                deprel_ext_path_edge=deprel_ext_path_edge
            )
        elif self.network_type == model.gat_type['gat_kb']:
            context, attn_per_head, next_relations = self.gat_kb(
                inp=inp,
                relation_inp=relations,
                dep_rel_pos_edge=dep_rel_pos_edge,
                word_rel_pos_edge=word_rel_pos_edge,
                deprel_edge=deprel_edge,
                deprel_ext_edge=deprel_ext_edge,
                deparc_edge=deparc_edge,
                edge_index=edge_index,
                dist_edge=dist_edge,
                deprel_path_edge=deprel_path_edge,
                deparc_path_edge=deparc_path_edge,
                path_len_edge=path_len_edge,
                deprel_ext_path_edge=deprel_ext_path_edge
            )
        elif self.network_type == model.gat_type['gaan']:
            context, attn_per_head = self.gaan(
                inp=inp,
                deprel_edge=deprel_edge,
                deparc_edge=deparc_edge,
                dep_rel_pos_edge=dep_rel_pos_edge,
                word_rel_pos_edge=word_rel_pos_edge,
                edge_index=edge_index,
                dist_edge=dist_edge,
                deprel_ext_edge=deprel_ext_edge,
                deprel_path_edge=deprel_path_edge,
                deparc_path_edge=deparc_path_edge,
                path_len_edge=path_len_edge,
                deprel_ext_path_edge=deprel_ext_path_edge
            )
        elif self.network_type == model.gat_type['san']:
            context, attn_per_head = self.san(
                inp=inp,
                deprel_edge=deprel_edge,
                deparc_edge=deparc_edge,
                dep_rel_pos_edge=dep_rel_pos_edge,
                word_rel_pos_edge=word_rel_pos_edge,
                edge_index=edge_index,
                dist_edge=dist_edge,
                deprel_ext_edge=deprel_ext_edge,
                deprel_path_edge=deprel_path_edge,
                deparc_path_edge=deparc_path_edge,
                path_len_edge=path_len_edge,
                deprel_ext_path_edge=deprel_ext_path_edge
            )
        elif self.network_type == model.gcn_type['gcn_syntactic']:
            context = self.gcn_syntactic(
                inp=inp,
                deprel_edge=deprel_edge,
                deparc_edge=deparc_edge,
                edge_index=edge_index
            )
        elif self.network_type == model.gcn_type['gcn_relational']:
            context = self.gcn_relational(
                inp=inp,
                deprel_ext_edge=deprel_ext_edge,
                edge_index=edge_index
            )
        elif self.network_type == model.gcn_type['gcn_ar']:
            context, attn_per_head = self.gcn_ar(
                inp=inp,
                dep_rel_pos_edge=dep_rel_pos_edge,
                word_rel_pos_edge=word_rel_pos_edge,
                deprel_ext_edge=deprel_ext_edge,
                edge_index=edge_index,
                deprel_edge=deprel_edge,
                deparc_edge=deparc_edge
            )
        else:
            raise Exception(f'GNN type is unknown. GNN type: {self.network_type}.')

        output = context

        if self.activation is not None:
            output = self.activation(context)

        if self.is_dropout_final:
            output = self.dropout(output)

        if self.is_residual:
            output = output + inp

        if self.activation is not None and self.is_layer_norm:
            output = self.layer_norm(output)

        return output, attn_per_head, next_relations
