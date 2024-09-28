import torch.nn as nn
import torch
from constants import word


class SyntacticGCN(nn.Module):
    def __init__(
        self,
        deparc_voc,
        deprel_size,
        model_dim,
        dropout,
        activation,
        init_weight_xavier_uniform
    ):
        super(SyntacticGCN, self).__init__()

        self.model_dim = model_dim
        self.align_idx = deparc_voc[word.deparc_map['align']]
        self.opposite_idx = deparc_voc[word.deparc_map['opposite']]
        self.self_idx = deparc_voc[word.deparc_map['self']]
        self.no_rel_idx = deparc_voc[word.deparc_map['norelation']]

        # Align
        self.V_in = nn.Parameter(torch.Tensor(self.model_dim, self.model_dim))
        self.b_in = nn.Parameter(torch.Tensor(deprel_size, self.model_dim))

        self.V_in_gate = nn.Parameter(torch.Tensor(self.model_dim, 1))
        self.b_in_gate = nn.Parameter(torch.Tensor(deprel_size, 1))

        # Opposite
        self.V_out = nn.Parameter(torch.Tensor(self.model_dim, self.model_dim))
        self.b_out = nn.Parameter(torch.Tensor(deprel_size, self.model_dim))

        self.V_out_gate = nn.Parameter(torch.Tensor(self.model_dim, 1))
        self.b_out_gate = nn.Parameter(torch.Tensor(deprel_size, 1))

        # Self
        self.W_self_loop = nn.Parameter(torch.Tensor(self.model_dim, self.model_dim))
        self.W_self_loop_gate = nn.Parameter(torch.Tensor(self.model_dim, 1))

        # No relation
        self.W_no_relation = nn.Parameter(torch.Tensor(self.model_dim, self.model_dim))
        self.W_no_relation_gate = nn.Parameter(torch.Tensor(self.model_dim, 1))

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.is_init_weight_xavier_uniform = init_weight_xavier_uniform

        self.init_params()

    def init_params(self):
        if self.is_init_weight_xavier_uniform:
            nn.init.xavier_uniform_(self.V_in, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.V_in_gate, gain=nn.init.calculate_gain('relu'))

            nn.init.xavier_uniform_(self.V_out, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.V_out_gate, gain=nn.init.calculate_gain('relu'))

            nn.init.xavier_uniform_(self.W_self_loop, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.W_self_loop_gate, gain=nn.init.calculate_gain('relu'))

            nn.init.xavier_uniform_(self.W_no_relation, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.W_no_relation_gate, gain=nn.init.calculate_gain('relu'))
        else:
            nn.init.xavier_normal_(self.V_in)
            nn.init.uniform_(self.V_in_gate)

            nn.init.xavier_normal_(self.V_out)
            nn.init.uniform_(self.V_out_gate)

            nn.init.xavier_normal_(self.W_self_loop)
            nn.init.uniform_(self.W_self_loop_gate)

            nn.init.xavier_normal_(self.W_no_relation)
            nn.init.uniform_(self.W_no_relation_gate)

        nn.init.zeros_(self.b_in)
        nn.init.ones_(self.b_in_gate)

        nn.init.zeros_(self.b_out)
        nn.init.ones_(self.b_out_gate)

    def sum_edge_scores_neighborhood_aware(
        self,
        scores_per_edge,
        tgt_index,
        neighbor_sum
    ):
        tgt_index_broadcasted = self.explicit_broadcast(tgt_index, scores_per_edge)

        neighbor_sum.scatter_add_(0, tgt_index_broadcasted, scores_per_edge)

    def forward(
        self,
        inp,
        deprel_edge,
        deparc_edge,
        edge_index
    ):
        num_nodes = inp.size(0)

        in_edge_index = edge_index[:, deparc_edge == self.align_idx]
        in_deprel_edge = deprel_edge[deparc_edge == self.align_idx]

        out_edge_index = edge_index[:, deparc_edge == self.opposite_idx]
        out_deprel_edge = deprel_edge[deparc_edge == self.opposite_idx]

        self_edge_index = edge_index[:, deparc_edge == self.self_idx]
        no_rel_edge_index = edge_index[:, deparc_edge == self.no_rel_idx]

        potential_in = None

        if in_edge_index.size(1) > 0:
            # node_size x model_dim
            input_in = torch.matmul(inp, self.V_in)
            # in_edge_size x model_dim
            first_in = input_in.index_select(0, in_edge_index[0])

            # in_edge_size x model_dim
            second_in = self.b_in.index_select(0, in_deprel_edge)
            in_ = first_in + second_in

            # node_size x 1
            input_in_gate = torch.matmul(inp, self.V_in_gate)
            # in_edge_size x 1
            first_in_gate = input_in_gate.index_select(0, in_edge_index[0])

            # in_edge_size x 1
            second_in_gate = self.b_in_gate.index_select(0, in_deprel_edge)
            in_gate = first_in_gate + second_in_gate
            potential_in_gate = self.sigmoid(in_gate)

            potential_in = in_ * potential_in_gate

        potential_out = None

        if out_edge_index.size(1) > 0:
            # node_size x model_dim
            input_out = torch.matmul(inp, self.V_out)
            # out_edge_size x model_dim
            first_out = input_out.index_select(0, out_edge_index[0])

            # out_edge_size x model_dim
            second_out = self.b_out.index_select(0, out_deprel_edge)
            out_ = first_out + second_out

            # node_size x 1
            input_out_gate = torch.matmul(inp, self.V_out_gate)
            # out_edge_size x 1
            first_out_gate = input_out_gate.index_select(0, out_edge_index[0])

            # out_edge_size x 1
            second_out_gate = self.b_out_gate.index_select(0, out_deprel_edge)
            out_gate = first_out_gate + second_out_gate
            potential_out_gate = self.sigmoid(out_gate)

            potential_out = out_ * potential_out_gate

        potential_same_input = None

        if self_edge_index.size(1) > 0:
            # node_size x model_dim
            same_input = torch.matmul(inp, self.W_self_loop)
            # self_edge_size x model_dim
            same_input = same_input.index_select(0, self_edge_index[0])

            # node_size x 1
            same_input_gate = torch.matmul(inp, self.W_self_loop_gate)
            # self_edge_size x 1
            same_input_gate = same_input_gate.index_select(0, self_edge_index[0])
            potential_same_input_gate = self.sigmoid(same_input_gate)

            potential_same_input = same_input * potential_same_input_gate

        potential_no_rel_input = None

        if no_rel_edge_index.size(1) > 0:
            # node_size x model_dim
            no_rel_input = torch.matmul(inp, self.W_no_relation)
            # no_rel_edge_size x model_dim
            no_rel_input = no_rel_input.index_select(0, no_rel_edge_index[0])

            # node_size x 1
            no_rel_input_gate = torch.matmul(inp, self.W_no_relation_gate)
            # no_rel_edge_size x 1
            no_rel_input_gate = no_rel_input_gate.index_select(0, no_rel_edge_index[0])
            potential_no_rel_input_gate = self.sigmoid(no_rel_input_gate)

            potential_no_rel_input = no_rel_input * potential_no_rel_input_gate

        assert potential_same_input is not None

        neighbor_sum = torch.zeros((num_nodes, self.model_dim)).to(potential_same_input)

        if potential_in:
            self.sum_edge_scores_neighborhood_aware(
                scores_per_edge=potential_in,
                tgt_index=in_edge_index[1],
                neighbor_sum=neighbor_sum
            )

        if potential_out:
            self.sum_edge_scores_neighborhood_aware(
                scores_per_edge=potential_out,
                tgt_index=out_edge_index[1],
                neighbor_sum=neighbor_sum
            )

        if potential_same_input:
            self.sum_edge_scores_neighborhood_aware(
                scores_per_edge=potential_same_input,
                tgt_index=self_edge_index[1],
                neighbor_sum=neighbor_sum
            )

        if potential_no_rel_input:
            self.sum_edge_scores_neighborhood_aware(
                scores_per_edge=potential_no_rel_input,
                tgt_index=no_rel_edge_index[1],
                neighbor_sum=neighbor_sum
            )

        # node_size x model_dim
        output = neighbor_sum

        output = output + inp

        if self.activation is not None:
            output = self.activation(output)

        output = self.dropout(output)

        return output
