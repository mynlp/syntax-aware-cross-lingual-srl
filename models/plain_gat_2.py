import torch
import torch.nn as nn
from constants import word
from utils import model


class PlainGAT(nn.Module):
    def __init__(
        self,
        deparc_voc,
        num_heads,
        model_dim,
        dropout,
        normalized,
        concat,
        final_bias,
        omit_self_edge,
        proj_self_edge,
        concat_input,
        apply_gated_head,
        neighbor_dim,
        bias
    ):
        super(PlainGAT, self).__init__()

        self.num_heads = num_heads
        self.model_dim = model_dim
        self.is_concat = concat
        self.is_final_bias = final_bias
        self.omit_self_edge = omit_self_edge
        self.proj_self_edge = proj_self_edge
        self.concat_input = concat_input
        self.apply_gated_head = apply_gated_head
        self.neighbor_dim = neighbor_dim
        self.is_bias = bias
        self.self_idx = deparc_voc[word.deparc_map['self']]

        if self.apply_gated_head:
            self.neighbor = nn.Linear(model_dim, neighbor_dim, bias=bias)

            if self.concat_input:
                self.gate = nn.Linear(2 * model_dim + neighbor_dim, self.num_heads, bias=bias)
            else:
                self.gate = nn.Linear(model_dim + neighbor_dim, self.num_heads, bias=bias)

        assert self.model_dim % self.num_heads == 0
        if self.is_concat:
            self.d_v = int(self.model_dim / self.num_heads)
        else:
            self.d_v = self.model_dim

        if self.is_final_bias:
            self.final_bias = nn.Parameter(torch.Tensor(self.model_dim))
        else:
            self.final_bias = None

        if self.proj_self_edge:
            self.self_weight = nn.Linear(self.model_dim, self.model_dim, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.value = nn.Linear(model_dim, num_heads * self.d_v, bias=False)
        self.weight_vector_value_src = nn.Parameter(torch.Tensor(1, self.num_heads, self.d_v))
        self.weight_vector_value_tgt = nn.Parameter(torch.Tensor(1, self.num_heads, self.d_v))

        self.dropout = nn.Dropout(dropout)
        self.normalized = normalized

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.weight_vector_value_src)
        nn.init.xavier_uniform_(self.weight_vector_value_tgt)

        if self.apply_gated_head:
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.xavier_uniform_(self.neighbor.weight)

        if self.proj_self_edge:
            nn.init.xavier_uniform_(self.self_weight.weight)

        if self.is_final_bias:
            nn.init.zeros_(self.final_bias)

        if self.is_bias:
            if self.apply_gated_head:
                nn.init.zeros_(self.gate.bias)
                nn.init.zeros_(self.neighbor.bias)

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
        edge_index,
        dist_edge,
        deparc_edge
    ):
        num_nodes = inp.size(0)

        if self.omit_self_edge:
            filtered_idx = deparc_edge != self.self_idx
            f_edge_index = edge_index[:, filtered_idx]
            f_dist_edge = dist_edge[filtered_idx]
        else:
            f_edge_index = edge_index
            f_dist_edge = dist_edge

        f_num_edges = f_edge_index.size(1)

        # 1) Project value.
        # node_size x nhead x d_v
        value = self.value(inp).view(-1, self.num_heads, self.d_v)

        # 2) Concatenate source and target node representations, multiply with weight vector, and apply leaky ReLU.
        src_index = f_edge_index[0]
        tgt_index = f_edge_index[1]

        # node_size x nhead
        value_src = (value * self.weight_vector_value_src).sum(dim=-1)
        value_tgt = (value * self.weight_vector_value_tgt).sum(dim=-1)

        # edge_size x nhead x d_v
        node_src = value.index_select(0, src_index)
        # edge_size x nhead
        scores_src = value_src.index_select(0, src_index)
        scores_tgt = value_tgt.index_select(0, tgt_index)
        scores = self.leakyReLU(scores_src + scores_tgt)

        # 3) Apply softmax over neighborhood to produce attention.
        scores = scores - scores.max()
        exp_scores = scores.exp()

        # node_size x nhead
        neighbor_sum = torch.zeros((num_nodes, self.num_heads)).to(exp_scores)
        self.sum_edge_scores_neighborhood_aware(
            scores_per_edge=exp_scores,
            tgt_index=tgt_index,
            neighbor_sum=neighbor_sum
        )

        # edge_size x nhead
        neighbor_aware_denominator = neighbor_sum.index_select(0, tgt_index)
        attn = exp_scores / (neighbor_aware_denominator + (1 / word.INFINITY_NUMBER))

        if self.normalized:
            # edge_size x 1
            unsq_dist_edge = f_dist_edge.unsqueeze(-1)
            # edge_size x nhead
            attn = attn / unsq_dist_edge
            neighbor_attn_sum = torch.zeros((num_nodes, self.num_heads)).to(attn)
            self.sum_edge_scores_neighborhood_aware(
                scores_per_edge=attn,
                tgt_index=tgt_index,
                neighbor_sum=neighbor_attn_sum
            )
            attn = attn / neighbor_attn_sum.index_select(0, tgt_index)

        # edge_size x nhead x 1
        attn = attn.unsqueeze(-1)
        attn = self.dropout(attn)

        # 4) Update final node representation.
        # edge_size x nhead x d_v
        weighted_node_src = node_src * attn
        # node_size x nhead x d_v
        output = torch.zeros((num_nodes, self.num_heads, self.d_v)).to(weighted_node_src)
        self.sum_edge_scores_neighborhood_aware(
            scores_per_edge=weighted_node_src,
            tgt_index=tgt_index,
            neighbor_sum=output
        )

        if self.apply_gated_head:
            # edge_size x model_dim
            neighbor = inp.index_select(0, src_index)
            # edge_size x neighbor_dim
            projected_neighbor = self.neighbor(inp).index_select(0, src_index)
            max_gate, avg_gate = model.apply_gate_to_weight_heads_in_gnn(
                neighbor=neighbor,
                projected_neighbor=projected_neighbor,
                num_nodes=num_nodes,
                model_dim=self.model_dim,
                tgt_index=tgt_index,
                num_edges=f_num_edges,
                neighbor_dim=self.neighbor_dim
            )

            if self.concat_input:
                # node_dim x nhead x 1
                gate = self.sigmoid(self.gate(torch.cat((inp, max_gate, avg_gate), dim=-1))).unsqueeze(-1)
            else:
                # node_dim x nhead x 1
                gate = self.sigmoid(self.gate(torch.cat((max_gate, avg_gate), dim=-1))).unsqueeze(-1)

            output = gate * output

        if self.is_concat:
            # node_size x model_dim (=num_heads x d_v)
            output = output.view(-1, self.num_heads * self.d_v)
        else:
            # node_size x model_dim (=d_v)
            output = output.mean(dim=1)

        if self.proj_self_edge:
            output = output + self.self_weight(inp)

        if self.is_final_bias:
            output = output + self.final_bias

        # a list of size num_heads containing tensors
        # of shape edge_size
        attn_per_head = [att.squeeze(1) for att in attn.squeeze(-1).chunk(self.num_heads, dim=1)]

        return output, attn_per_head
