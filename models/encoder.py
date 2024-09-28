import torch.nn as nn
from models.transformer_encoder import TransformerEncoder
from models.gnn_encoder import GNNEncoder
from models.lstm_encoder import LSTMEncoder
from utils.model import is_trans
from constants import model


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        if len(args.max_relative_positions) != args.num_enc_layers:
            assert len(args.max_relative_positions) == 1
            args.max_relative_positions = args.max_relative_positions * args.num_enc_layers

        if len(args.dep_max_relative_positions) != args.num_enc_layers:
            assert len(args.dep_max_relative_positions) == 1
            args.dep_max_relative_positions = args.dep_max_relative_positions * args.num_enc_layers

        self.network_types = args.network_type
        d_model = args.hid_dim

        for idx, network_type in enumerate(self.network_types):
            is_tr = is_trans([network_type])

            written_idx = '' if idx == 0 else f'{idx}'

            if is_tr:
                setattr(self, f'encoder{written_idx}', TransformerEncoder(
                    num_layers=args.num_enc_layers,
                    d_model=d_model,
                    num_heads=args.num_heads,
                    d_k=args.d_k,
                    d_v=args.d_v,
                    d_ff=args.d_ff,
                    dropout=args.enc_dropout,
                    max_relative_positions=args.max_relative_positions,
                    dep_max_relative_positions=args.dep_max_relative_positions,
                    use_neg_dist=args.use_neg_dist,
                    use_deprel=args.use_deprel,
                    use_deprel_ext=args.use_deprel_ext,
                    lstm_dropout=args.lstm_dropout,
                    use_dep_path=args.use_dep_path,
                    use_dep_ext_path=args.use_dep_ext_path,
                    deprel_size=args.deprel_size,
                    deparc_size=args.deparc_size,
                    deprel_ext_size=args.deprel_ext_size,
                    deprel_edge_dim=args.deprel_edge_dim,
                    deparc_edge_dim=args.deparc_edge_dim,
                    normalized=args.normalized,
                    bias=args.bias,
                    sum_dep_path=args.sum_dep_path,
                    network_type=network_type,
                    neighbor_dim=args.neighbor_dim,
                    omit_self_edge=args.omit_self_edge,
                    proj_self_edge=args.proj_self_edge,
                    concat_input=args.concat_input,
                    apply_gated_head=args.apply_gated_head,
                    use_positional_embedding=args.use_positional_embedding,
                    use_dep_rel_pos=args.use_dep_rel_pos,
                    use_word_rel_pos=args.use_word_rel_pos
                ))
            elif network_type == model.network_type['lstm']:
                if args.lstm_activation == 'elu':
                    activation_function = nn.ELU()
                elif args.lstm_activation == 'leaky_relu':
                    activation_function = nn.LeakyReLU()
                elif args.lstm_activation == 'relu':
                    activation_function = nn.ReLU()
                else:
                    activation_function = None

                setattr(self, f'encoder{written_idx}', LSTMEncoder(
                    input_size=d_model,
                    hidden_size=args.lstm_hidden_size,
                    num_layers=args.lstm_num_layers,
                    dropout=args.lstm_dropout_net,
                    bidirectional=True,
                    activation=activation_function,
                    dropout_final=args.lstm_dropout_net_final,
                    residual=args.lstm_residual,
                    layer_norm=args.lstm_layer_norm
                ))

                d_model = getattr(self, f'encoder{written_idx}').out_size
            else:
                setattr(self, f'encoder{written_idx}', GNNEncoder(
                    num_layers=args.num_enc_layers,
                    d_model=d_model,
                    num_heads=args.num_heads,
                    dropout=args.enc_dropout,
                    max_relative_positions=args.max_relative_positions,
                    use_neg_dist=args.use_neg_dist,
                    use_deprel=args.use_deprel,
                    use_deprel_ext=args.use_deprel_ext,
                    lstm_dropout=args.lstm_dropout,
                    use_dep_path=args.use_dep_path,
                    use_dep_ext_path=args.use_dep_ext_path,
                    deprel_size=args.deprel_size,
                    deprel_ext_size=args.deprel_ext_size,
                    deparc_size=args.deparc_size,
                    deprel_edge_dim=args.deprel_edge_dim,
                    deparc_edge_dim=args.deparc_edge_dim,
                    normalized=args.normalized,
                    bias=args.bias,
                    init_weight_xavier_uniform=args.init_weight_xavier_uniform,
                    concat_at_final_layer=args.gnn_concat_at_final_layer,
                    activation_at_final_layer=args.gnn_activation_at_final_layer,
                    activation=args.gnn_activation,
                    network_type=network_type,
                    dropout_final=args.gnn_dropout_final,
                    residual=args.gnn_residual,
                    layer_norm=args.gnn_layer_norm,
                    final_bias=args.gnn_final_bias,
                    deparc_voc=args.deparc_voc,
                    deprel_ext_voc=args.deprel_ext_voc,
                    base_size=args.base_size,
                    use_positional_embedding=args.use_positional_embedding,
                    deprel_ext_edge_dim=args.deprel_ext_edge_dim,
                    rel_pos_dim=args.rel_pos_dim,
                    att_dim=args.att_dim,
                    sum_dep_path=args.sum_dep_path,
                    omit_self_edge=args.omit_self_edge,
                    proj_self_edge=args.proj_self_edge,
                    d_k=args.d_k,
                    d_v=args.d_v,
                    neighbor_dim=args.neighbor_dim,
                    concat_input=args.concat_input,
                    apply_gated_head=args.apply_gated_head,
                    same_relation_dim=args.gnn_same_relation_dim,
                    use_dep_rel_pos=args.use_dep_rel_pos,
                    use_word_rel_pos=args.use_word_rel_pos,
                    average_heads=args.gnn_average_heads
                ))

    def count_parameters(self):
        total = 0

        for idx, network_type in enumerate(self.network_types):
            written_idx = '' if idx == 0 else f'{idx}'

            encoder = getattr(self, f'encoder{written_idx}')
            total += encoder.count_parameters()

        return total

    def forward(
            self,
            **kwargs,
    ):
        out = kwargs.get('inp')
        sent_len_rep = kwargs.get('sent_len_rep')

        for idx, network_type in enumerate(self.network_types):
            is_tr = is_trans([network_type])

            written_idx = '' if idx == 0 else f'{idx}'

            encoder = getattr(self, f'encoder{written_idx}')

            if is_tr:
                adj_mask_rep = kwargs.get('adj_mask_rep')
                fc_mask_rep = kwargs.get('fc_mask_rep')
                dep_rel_pos_mat_rep = kwargs.get('dep_rel_pos_mat_rep')
                word_rel_pos_mat_rep = kwargs.get('word_rel_pos_mat_rep')
                deprel_mat_rep = kwargs.get('deprel_mat_rep')
                deparc_mat_rep = kwargs.get('deparc_mat_rep')
                path_len_mat_rep = kwargs.get('path_len_mat_rep')
                deprel_path_mat_rep = kwargs.get('deprel_path_mat_rep')
                deparc_path_mat_rep = kwargs.get('deparc_path_mat_rep')
                deprel_ext_mat_rep = kwargs.get('deprel_ext_mat_rep')
                deprel_ext_path_mat_rep = kwargs.get('deprel_ext_path_mat_rep')

                layer_outputs, _ = encoder(
                    inp=out,
                    sent_len_rep=sent_len_rep,
                    adj_mask_rep=adj_mask_rep,
                    dep_rel_pos_mat_rep=dep_rel_pos_mat_rep,
                    word_rel_pos_mat_rep=word_rel_pos_mat_rep,
                    fc_mask_rep=fc_mask_rep,
                    deprel_mat_rep=deprel_mat_rep,
                    deparc_mat_rep=deparc_mat_rep,
                    path_len_mat_rep=path_len_mat_rep,
                    deprel_path_mat_rep=deprel_path_mat_rep,
                    deparc_path_mat_rep=deparc_path_mat_rep,
                    deprel_ext_mat_rep=deprel_ext_mat_rep,
                    deprel_ext_path_mat_rep=deprel_ext_path_mat_rep
                )  # B x seq_len x h
            elif network_type == model.network_type['lstm']:
                output = encoder(
                    inp=out
                )  # B x seq_len x h
                layer_outputs = [output]
            else:
                dist_edge_rep = kwargs.get('dist_edge_rep')
                edge_len_rep = kwargs.get('edge_len_rep')
                edge_index_rep = kwargs.get('edge_index_rep')
                dep_rel_pos_edge_rep = kwargs.get('dep_rel_pos_edge_rep')
                word_rel_pos_edge_rep = kwargs.get('word_rel_pos_edge_rep')
                deprel_edge_rep = kwargs.get('deprel_edge_rep')
                deparc_edge_rep = kwargs.get('deparc_edge_rep')
                deprel_path_edge_rep = kwargs.get('deprel_path_edge_rep')
                deparc_path_edge_rep = kwargs.get('deparc_path_edge_rep')
                path_len_edge_rep = kwargs.get('path_len_edge_rep')
                deprel_ext_edge_rep = kwargs.get('deprel_ext_edge_rep')
                deprel_ext_path_edge_rep = kwargs.get('deprel_ext_path_edge_rep')

                layer_outputs, _ = encoder(
                    inp=out,
                    sent_len_rep=sent_len_rep,
                    edge_len_rep=edge_len_rep,
                    edge_index_rep=edge_index_rep,
                    dep_rel_pos_edge_rep=dep_rel_pos_edge_rep,
                    word_rel_pos_edge_rep=word_rel_pos_edge_rep,
                    deprel_edge_rep=deprel_edge_rep,
                    deparc_edge_rep=deparc_edge_rep,
                    deprel_path_edge_rep=deprel_path_edge_rep,
                    deparc_path_edge_rep=deparc_path_edge_rep,
                    path_len_edge_rep=path_len_edge_rep,
                    dist_edge_rep=dist_edge_rep,
                    deprel_ext_edge_rep=deprel_ext_edge_rep,
                    deprel_ext_path_edge_rep=deprel_ext_path_edge_rep
                )  # B x seq_len x h

            out = layer_outputs[-1]

        return out, layer_outputs
