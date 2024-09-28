import torch
import torch.nn as nn
from constants import word
from models.positional_embedding import PositionalEmbedding
from utils import model


class ARGCN(nn.Module):
    def __init__(
        self,
        deprel_ext_size,
        deprel_ext_voc,
        use_neg_dist,
        max_relative_position,
        use_positional_embedding,
        deprel_ext_edge_dim,
        use_deprel_ext,
        rel_pos_dim,
        model_dim,
        att_dim,
        num_heads,
        dropout,
        final_bias,
        omit_self_edge,
        proj_self_edge,
        concat_input,
        apply_gated_head,
        neighbor_dim,
        bias,
        d_v,
        use_dep_rel_pos,
        use_word_rel_pos,
        use_deprel,
        deprel_size,
        deparc_size,
        deprel_edge_dim,
        deparc_edge_dim
    ):
        super(ARGCN, self).__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.att_dim = att_dim
        self.is_final_bias = final_bias
        self.max_relative_position = max_relative_position
        self.use_neg_dist = use_neg_dist
        self.omit_self_edge = omit_self_edge
        self.proj_self_edge = proj_self_edge
        self.concat_input = concat_input
        self.apply_gated_head = apply_gated_head
        self.neighbor_dim = neighbor_dim
        self.is_bias = bias
        self.use_dep_rel_pos = use_dep_rel_pos
        self.use_word_rel_pos = use_word_rel_pos
        self.use_deprel_ext = use_deprel_ext
        self.use_deprel = use_deprel

        self.self_idx = deprel_ext_voc[word.SELF_DEPREL]
        self.d_v = d_v

        if self.concat_input:
            self.final_weight = nn.Linear(model_dim + (self.d_v * self.num_heads), self.model_dim, bias=False)
        else:
            self.final_weight = nn.Linear(self.d_v * self.num_heads, self.model_dim, bias=False)

        if self.apply_gated_head:
            self.neighbor = nn.Linear(model_dim, neighbor_dim, bias=bias)
            if self.concat_input:
                self.gate = nn.Linear(2 * model_dim + neighbor_dim, self.num_heads, bias=bias)
            else:
                self.gate = nn.Linear(model_dim + neighbor_dim, self.num_heads, bias=bias)

        self.value_weight = nn.Linear(self.model_dim, self.d_v * self.num_heads, bias=False)
        self.key_query_weight = nn.Linear(self.model_dim, self.d_v * self.num_heads, bias=False)
        self.att_weight = nn.Parameter(torch.Tensor(self.num_heads, (self.d_v * 2) + rel_pos_dim, self.att_dim))
        if self.use_deprel:
            self.edge_weight = nn.Parameter(torch.Tensor(self.num_heads, deprel_edge_dim + deparc_edge_dim + self.att_dim, 1))
        else:
            self.edge_weight = nn.Parameter(torch.Tensor(self.num_heads, deprel_ext_edge_dim + self.att_dim, 1))

        if self.proj_self_edge:
            self.self_weight = nn.Linear(self.model_dim, self.model_dim, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        if self.use_deprel:
            self.deprel_embeddings = nn.Embedding(
                deprel_size,
                deprel_edge_dim,
                padding_idx=word.PAD
            )
            self.deparc_embeddings = nn.Embedding(
                deparc_size,
                deparc_edge_dim,
                padding_idx=word.PAD
            )
        else:
            self.deprel_ext_embeddings = nn.Embedding(
                deprel_ext_size,
                deprel_ext_edge_dim,
                padding_idx=word.PAD
            )

        assert self.max_relative_position > 0

        vocab_size = self.max_relative_position * 2 + 1 \
            if self.use_neg_dist else self.max_relative_position + 1

        if use_positional_embedding:
            self.relative_positions_embeddings = PositionalEmbedding(rel_pos_dim)
        else:
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size,
                rel_pos_dim
            )

        if self.is_final_bias:
            self.final_bias = nn.Parameter(torch.Tensor(self.model_dim))

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.value_weight.weight)
        nn.init.xavier_uniform_(self.key_query_weight.weight)
        nn.init.xavier_uniform_(self.final_weight.weight)
        nn.init.xavier_uniform_(self.att_weight)
        nn.init.xavier_uniform_(self.edge_weight)

        if self.apply_gated_head:
            nn.init.xavier_uniform_(self.neighbor.weight)
            nn.init.xavier_uniform_(self.gate.weight)

        if self.proj_self_edge:
            nn.init.xavier_uniform_(self.self_weight.weight)

        if self.is_final_bias:
            nn.init.zeros_(self.final_bias)

        if self.is_bias:
            if self.apply_gated_head:
                nn.init.zeros_(self.neighbor.bias)
                nn.init.zeros_(self.gate.bias)

    def sum_edge_scores_neighborhood_aware(
        self,
        scores_per_edge,
        tgt_index,
        neighbor_sum
    ):
        tgt_index_broadcasted = self.explicit_broadcast(tgt_index, scores_per_edge)

        neighbor_sum.scatter_add_(0, tgt_index_broadcasted, scores_per_edge)

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def forward(
        self,
        inp,
        dep_rel_pos_edge,
        word_rel_pos_edge,
        deprel_ext_edge,
        edge_index,
        deprel_edge,
        deparc_edge
    ):
        num_nodes = inp.size(0)

        if self.omit_self_edge:
            filtered_idx = deprel_ext_edge != self.self_idx
            f_edge_index = edge_index[:, filtered_idx]
            f_deprel_ext_edge = deprel_ext_edge[filtered_idx]
            f_deprel_edge = deprel_edge[filtered_idx]
            f_deparc_edge = deparc_edge[filtered_idx]
            if self.use_word_rel_pos:
                f_rel_pos_edge = word_rel_pos_edge[filtered_idx]
            else:
                f_rel_pos_edge = dep_rel_pos_edge[filtered_idx]
        else:
            f_edge_index = edge_index
            f_deprel_ext_edge = deprel_ext_edge
            f_deprel_edge = deprel_edge
            f_deparc_edge = deparc_edge
            if self.use_word_rel_pos:
                f_rel_pos_edge = word_rel_pos_edge
            else:
                f_rel_pos_edge = dep_rel_pos_edge

        f_num_edges = f_edge_index.size(1)

        if self.use_deprel:
            deprel = self.deprel_embeddings(f_deprel_edge)
            deparc = self.deparc_embeddings(f_deparc_edge)

            b_r = torch.cat((deprel, deparc), -1).unsqueeze(1).repeat(1, self.num_heads, 1)
        else:
            # edge_size x nhead x deprel_ext_edge_dim
            b_r = self.deprel_ext_embeddings(f_deprel_ext_edge).unsqueeze(1).repeat(1, self.num_heads, 1)
            # edge_size x nhead x rel_pos_dim

        if self.use_neg_dist:
            f_rel_pos_edge = f_rel_pos_edge + self.max_relative_position
        else:
            f_rel_pos_edge = torch.abs(f_rel_pos_edge)

        p = self.relative_positions_embeddings(f_rel_pos_edge).unsqueeze(1).repeat(
            1,
            self.num_heads,
            1
        )

        src_index = f_edge_index[0]
        tgt_index = f_edge_index[1]

        kq_h = self.key_query_weight(inp).view(-1, self.num_heads, self.d_v)

        # edge_size x nhead x d_v
        kq_h_src = kq_h.index_select(0, src_index)

        # edge_size x nhead x d_v
        kq_h_tgt = kq_h.index_select(0, tgt_index)

        # edge_size x nhead x d_v
        v_h_src = self.value_weight(inp)\
            .view(-1, self.num_heads, self.d_v)\
            .index_select(0, src_index)

        # edge_size x nhead x (d_v * 2 + rel_pos_dim)
        att_cat = torch.cat([kq_h_src, kq_h_tgt, p], dim=-1)

        # edge_size x nhead x att_dim
        att_scores = torch.einsum('eij,ijk->eik', att_cat, self.att_weight)
        att_scores = self.leakyReLU(att_scores)
        att_scores = att_scores - att_scores.max()
        exp_att_scores = att_scores.exp()
        neighbor_sum = torch.zeros((num_nodes, self.num_heads, self.att_dim)).to(exp_att_scores)
        self.sum_edge_scores_neighborhood_aware(
            scores_per_edge=exp_att_scores,
            tgt_index=tgt_index,
            neighbor_sum=neighbor_sum
        )
        neighbor_aware_denominator = neighbor_sum.index_select(0, tgt_index)
        beta = exp_att_scores / (neighbor_aware_denominator + (1 / word.INFINITY_NUMBER))

        # edge_size x nhead x att_dim
        # beta = self.dropout(beta)

        # edge_size x nhead x (deprel_ext_edge_dim + att_dim)
        edge_cat = torch.cat([b_r, beta], dim=-1)
        # edge_size x nhead x 1
        edge_scores = torch.einsum('eij,ijk->eik', edge_cat, self.edge_weight)
        edge_scores = self.leakyReLU(edge_scores)
        edge_scores = self.dropout(edge_scores)

        # edge_size x nhead x d_v
        weighted_v_h_src = v_h_src * edge_scores
        output = torch.zeros((num_nodes, self.num_heads, self.d_v)).to(weighted_v_h_src)
        self.sum_edge_scores_neighborhood_aware(
            scores_per_edge=weighted_v_h_src,
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

            # node_size x nhead x d_v
            output = gate * output

        output = output.view(-1, self.num_heads * self.d_v)

        if self.concat_input:
            output = self.final_weight(torch.cat((inp, output), dim=-1))
        else:
            output = self.final_weight(output)

        if self.proj_self_edge:
            output = output + self.self_weight(inp)

        if self.is_final_bias:
            output = output + self.final_bias

        attn_per_head = [att.squeeze(1) for att in edge_scores.squeeze(-1).chunk(self.num_heads, dim=1)]

        return output, attn_per_head
