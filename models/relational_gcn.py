import torch
import torch.nn as nn
from constants import word


class RelationalGCN(nn.Module):
    def __init__(
        self,
        deprel_ext_size,
        deprel_ext_voc,
        base_size,
        model_dim,
        final_bias,
        use_deprel_ext,
        omit_self_edge,
        proj_self_edge
    ):
        super(RelationalGCN, self).__init__()

        assert use_deprel_ext
        self.base_size = base_size
        self.deprel_ext_size = deprel_ext_size
        self.model_dim = model_dim
        self.is_final_bias = final_bias
        self.omit_self_edge = omit_self_edge
        self.proj_self_edge = proj_self_edge

        self.self_idx = deprel_ext_voc[word.SELF_DEPREL]

        if self.base_size <= 0 or self.base_size > self.deprel_ext_size:
            self.base_size = self.deprel_ext_size

        if self.proj_self_edge:
            self.self_weight = nn.Linear(self.model_dim, self.model_dim, bias=False)

        self.weight = nn.Parameter(torch.Tensor(
            self.base_size,
            self.model_dim,
            self.model_dim
        ))

        if self.base_size < self.deprel_ext_size:
            self.w_comp = nn.Parameter(torch.Tensor(
                self.deprel_ext_size,
                self.base_size
            ))

        if self.is_final_bias:
            self.final_bias = nn.Parameter(torch.Tensor(self.model_dim))

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.weight)

        if self.proj_self_edge:
            nn.init.xavier_uniform_(self.self_weight.weight)

        if self.base_size < self.deprel_ext_size:
            nn.init.xavier_uniform_(self.w_comp)

        if self.is_final_bias:
            nn.init.zeros_(self.final_bias)

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
        deprel_ext_edge,
        edge_index
    ):
        num_nodes = inp.size(0)

        if self.omit_self_edge:
            filtered_idx = deprel_ext_edge != self.self_idx
            f_edge_index = edge_index[:, filtered_idx]
            f_deprel_ext_edge = deprel_ext_edge[filtered_idx]
        else:
            f_edge_index = edge_index
            f_deprel_ext_edge = deprel_ext_edge

        f_edge_size = f_edge_index.size(1)

        if self.base_size < self.deprel_ext_size:
            weight = self.weight.view(self.base_size, self.model_dim * self.model_dim)
            weight = torch.matmul(self.w_comp, weight).view(self.deprel_ext_size, self.model_dim, self.model_dim)
        else:
            weight = self.weight

        src_index = f_edge_index[0]
        tgt_index = f_edge_index[1]

        # edge_size x model_dim x model_dim
        filtered_weight = weight.index_select(0, f_deprel_ext_edge)
        # edge_size x model_dim
        filtered_input = inp.index_select(0, src_index)
        # edge_size x model_dim
        potential_input = torch.einsum('ei,eij->ej', filtered_input, filtered_weight)

        edge_to_deprel = torch.zeros((f_edge_size, self.deprel_ext_size)).to(potential_input)
        edge_to_deprel[torch.arange(f_edge_size), f_deprel_ext_edge] = 1
        neighbor_count_acc = torch.zeros((num_nodes, self.deprel_ext_size)).to(potential_input)
        self.sum_edge_scores_neighborhood_aware(
            scores_per_edge=edge_to_deprel,
            tgt_index=tgt_index,
            neighbor_sum=neighbor_count_acc
        )
        # edge_size x 1
        denumerator = neighbor_count_acc[tgt_index, f_deprel_ext_edge].unsqueeze(-1)
        potential_input = potential_input / denumerator

        output = torch.zeros((num_nodes, self.model_dim)).to(potential_input)
        self.sum_edge_scores_neighborhood_aware(
            scores_per_edge=potential_input,
            tgt_index=tgt_index,
            neighbor_sum=output
        )

        if self.proj_self_edge:
            output = output + self.self_weight(inp)

        if self.is_final_bias:
            output = output + self.final_bias

        return output
