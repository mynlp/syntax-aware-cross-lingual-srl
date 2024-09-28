import torch.nn as nn
import torch
from constants import word


class SyntacticGCN(nn.Module):
    def __init__(
        self,
        deparc_voc,
        deprel_size,
        model_dim,
        use_deprel,
        init_weight_xavier_uniform
    ):
        super(SyntacticGCN, self).__init__()

        self.model_dim = model_dim
        self.align_idx = deparc_voc[word.deparc_map['align']]
        self.opposite_idx = deparc_voc[word.deparc_map['opposite']]
        self.self_idx = deparc_voc[word.deparc_map['self']]
        self.no_rel_idx = deparc_voc[word.deparc_map['norelation']]
        assert use_deprel

        # Align
        self.V_in = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.b_in = nn.Parameter(torch.Tensor(deprel_size, self.model_dim))

        self.V_in_gate = nn.Linear(self.model_dim, 1, bias=False)
        self.b_in_gate = nn.Parameter(torch.Tensor(deprel_size, 1))

        # Opposite
        self.V_out = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.b_out = nn.Parameter(torch.Tensor(deprel_size, self.model_dim))

        self.V_out_gate = nn.Linear(self.model_dim, 1, bias=False)
        self.b_out_gate = nn.Parameter(torch.Tensor(deprel_size, 1))

        # Self
        self.W_self_loop = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.W_self_loop_gate = nn.Linear(self.model_dim, 1, bias=False)

        # No relation
        self.W_no_relation = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.W_no_relation_gate = nn.Linear(self.model_dim, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.is_init_weight_xavier_uniform = init_weight_xavier_uniform

        self.init_params()

    def init_params(self):
        if self.is_init_weight_xavier_uniform:
            nn.init.xavier_uniform_(self.V_in.weight)
            nn.init.xavier_uniform_(self.V_in_gate.weight)

            nn.init.xavier_uniform_(self.V_out.weight)
            nn.init.xavier_uniform_(self.V_out_gate.weight)

            nn.init.xavier_uniform_(self.W_self_loop.weight)
            nn.init.xavier_uniform_(self.W_self_loop_gate.weight)

            nn.init.xavier_uniform_(self.W_no_relation.weight)
            nn.init.xavier_uniform_(self.W_no_relation_gate.weight)
        else:
            nn.init.xavier_normal_(self.V_in.weight)
            nn.init.uniform_(self.V_in_gate.weight)

            nn.init.xavier_normal_(self.V_out.weight)
            nn.init.uniform_(self.V_out_gate.weight)

            nn.init.xavier_normal_(self.W_self_loop.weight)
            nn.init.uniform_(self.W_self_loop_gate.weight)

            nn.init.xavier_normal_(self.W_no_relation.weight)
            nn.init.uniform_(self.W_no_relation_gate.weight)

        nn.init.zeros_(self.b_in)
        nn.init.ones_(self.b_in_gate)

        nn.init.zeros_(self.b_out)
        nn.init.ones_(self.b_out_gate)

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

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

        is_in = in_edge_index.size(1) > 0
        is_out = out_edge_index.size(1) > 0
        is_self = self_edge_index.size(1) > 0
        is_no_rel = no_rel_edge_index.size(1) > 0

        potential_in = None

        if is_in:
            # node_size x model_dim
            input_in = self.V_in(inp)
            # in_edge_size x model_dim
            first_in = input_in.index_select(0, in_edge_index[0])

            # in_edge_size x model_dim
            second_in = self.b_in.index_select(0, in_deprel_edge)
            in_ = first_in + second_in

            # node_size x 1
            input_in_gate = self.V_in_gate(inp)
            # in_edge_size x 1
            first_in_gate = input_in_gate.index_select(0, in_edge_index[0])

            # in_edge_size x 1
            second_in_gate = self.b_in_gate.index_select(0, in_deprel_edge)
            in_gate = first_in_gate + second_in_gate
            potential_in_gate = self.sigmoid(in_gate)

            potential_in = in_ * potential_in_gate

        potential_out = None

        if is_out:
            # node_size x model_dim
            input_out = self.V_out(inp)
            # out_edge_size x model_dim
            first_out = input_out.index_select(0, out_edge_index[0])

            # out_edge_size x model_dim
            second_out = self.b_out.index_select(0, out_deprel_edge)
            out_ = first_out + second_out

            # node_size x 1
            input_out_gate = self.V_out_gate(inp)
            # out_edge_size x 1
            first_out_gate = input_out_gate.index_select(0, out_edge_index[0])

            # out_edge_size x 1
            second_out_gate = self.b_out_gate.index_select(0, out_deprel_edge)
            out_gate = first_out_gate + second_out_gate
            potential_out_gate = self.sigmoid(out_gate)

            potential_out = out_ * potential_out_gate

        potential_same_input = None

        if is_self:
            # node_size x model_dim
            same_input = self.W_self_loop(inp)
            # self_edge_size x model_dim
            same_input = same_input.index_select(0, self_edge_index[0])

            # node_size x 1
            same_input_gate = self.W_self_loop_gate(inp)
            # self_edge_size x 1
            same_input_gate = same_input_gate.index_select(0, self_edge_index[0])
            potential_same_input_gate = self.sigmoid(same_input_gate)

            potential_same_input = same_input * potential_same_input_gate

        potential_no_rel_input = None

        if is_no_rel:
            # node_size x model_dim
            no_rel_input = self.W_no_relation(inp)
            # no_rel_edge_size x model_dim
            no_rel_input = no_rel_input.index_select(0, no_rel_edge_index[0])

            # node_size x 1
            no_rel_input_gate = self.W_no_relation_gate(inp)
            # no_rel_edge_size x 1
            no_rel_input_gate = no_rel_input_gate.index_select(0, no_rel_edge_index[0])
            potential_no_rel_input_gate = self.sigmoid(no_rel_input_gate)

            potential_no_rel_input = no_rel_input * potential_no_rel_input_gate

        assert potential_same_input is not None

        neighbor_sum = torch.zeros((num_nodes, self.model_dim)).to(potential_same_input)

        if is_in:
            self.sum_edge_scores_neighborhood_aware(
                scores_per_edge=potential_in,
                tgt_index=in_edge_index[1],
                neighbor_sum=neighbor_sum
            )

        if is_out:
            self.sum_edge_scores_neighborhood_aware(
                scores_per_edge=potential_out,
                tgt_index=out_edge_index[1],
                neighbor_sum=neighbor_sum
            )

        if is_self:
            self.sum_edge_scores_neighborhood_aware(
                scores_per_edge=potential_same_input,
                tgt_index=self_edge_index[1],
                neighbor_sum=neighbor_sum
            )

        if is_no_rel:
            self.sum_edge_scores_neighborhood_aware(
                scores_per_edge=potential_no_rel_input,
                tgt_index=no_rel_edge_index[1],
                neighbor_sum=neighbor_sum
            )

        # node_size x model_dim
        output = neighbor_sum

        return output
