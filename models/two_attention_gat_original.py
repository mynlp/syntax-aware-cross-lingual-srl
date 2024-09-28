import torch
import torch.nn as nn
from constants import word
from utils import model
from models.positional_embedding import PositionalEmbedding
from models.edge_dependency_path_encoder import EdgeDependencyPathEncoder


class TwoAttentionGATOriginal(nn.Module):
    def __init__(
        self,
        deparc_voc,
        deprel_ext_size,
        num_heads,
        model_dim,
        dropout,
        max_relative_position,
        use_neg_dist,
        use_deprel,
        use_deprel_ext,
        deprel_size,
        deparc_size,
        deprel_edge_dim,
        deparc_edge_dim,
        normalized,
        concat,
        rel_pos_dim,
        use_dep_path,
        use_dep_ext_path,
        omit_self_edge,
        proj_self_edge,
        concat_input,
        apply_gated_head,
        neighbor_dim,
        bias,
        use_positional_embedding,
        same_relation_dim,
        final_bias,
        deprel_ext_edge_dim,
        use_dep_rel_pos,
        use_word_rel_pos,
        lstm_dropout,
        sum_dep_path
    ):
        super(TwoAttentionGATOriginal, self).__init__()

        self.concat_input = concat_input
        self.apply_gated_head = apply_gated_head
        self.neighbor_dim = neighbor_dim
        self.is_bias = bias
        self.is_final_bias = final_bias
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.is_concat = concat
        self.max_relative_position = max_relative_position
        self.use_neg_dist = use_neg_dist
        self.use_deprel = use_deprel
        self.use_deprel_ext = use_deprel_ext
        self.normalized = normalized
        self.rel_pos_dim = rel_pos_dim
        self.use_dep_path = use_dep_path
        self.use_dep_ext_path = use_dep_ext_path
        self.omit_self_edge = omit_self_edge
        self.proj_self_edge = proj_self_edge
        self.self_idx = deparc_voc[word.deparc_map['self']]
        self.use_positional_embedding = use_positional_embedding
        self.use_dep_rel_pos = use_dep_rel_pos
        self.use_word_rel_pos = use_word_rel_pos

        assert self.model_dim % self.num_heads == 0
        assert (deprel_edge_dim + deparc_edge_dim) % self.num_heads == 0
        assert self.rel_pos_dim % self.num_heads == 0
        assert deprel_ext_edge_dim % self.num_heads == 0

        self.relation_dim = 0

        if self.max_relative_position > 0:
            if self.use_dep_rel_pos:
                if self.use_positional_embedding:
                    self.dep_relative_positions_embeddings = PositionalEmbedding(
                        self.rel_pos_dim
                    )
                else:
                    vocab_size = self.max_relative_position * 2 + 1 \
                        if self.use_neg_dist else self.max_relative_position + 1
                    self.dep_relative_positions_embeddings = nn.Embedding(
                        vocab_size,
                        self.rel_pos_dim
                    )
                self.relation_dim += self.rel_pos_dim
            if self.use_word_rel_pos:
                if self.use_positional_embedding:
                    self.word_relative_positions_embeddings = PositionalEmbedding(
                        self.rel_pos_dim
                    )
                else:
                    vocab_size = self.max_relative_position * 2 + 1 \
                        if self.use_neg_dist else self.max_relative_position + 1
                    self.word_relative_positions_embeddings = nn.Embedding(
                        vocab_size,
                        self.rel_pos_dim
                    )
                self.relation_dim += self.rel_pos_dim
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
            self.relation_dim += (deprel_edge_dim + deparc_edge_dim)
        elif self.use_deprel_ext:
            self.deprel_ext_embeddings = nn.Embedding(
                deprel_ext_size,
                deprel_ext_edge_dim,
                padding_idx=word.PAD
            )
            self.relation_dim += deprel_ext_edge_dim
        elif self.use_dep_path or self.use_dep_ext_path:
            if self.use_dep_ext_path:
                input_size = deprel_ext_edge_dim
            else:
                input_size = (deprel_edge_dim + deparc_edge_dim)

            self.dep_path_embeddings = EdgeDependencyPathEncoder(
                input_size=input_size,
                hidden_size=input_size,
                dropout=lstm_dropout,
                deprel_size=deprel_size,
                deparc_size=deparc_size,
                deprel_ext_size=deprel_ext_size,
                deprel_edge_dim=deprel_edge_dim,
                deparc_edge_dim=deparc_edge_dim,
                deprel_ext_edge_dim=deprel_ext_edge_dim,
                use_dep_path=use_dep_path,
                use_dep_ext_path=use_dep_ext_path,
                sum_dep_path=sum_dep_path
            )

            self.relation_dim += input_size

        if self.is_concat:
            self.d_v = int(self.model_dim / self.num_heads)

            if same_relation_dim:
                self.relation_d_v = self.relation_dim
            else:
                self.relation_d_v = int(self.relation_dim / self.num_heads)
        else:
            self.d_v = self.model_dim
            self.relation_d_v = self.relation_dim

        if self.proj_self_edge:
            self.self_weight = nn.Linear(self.model_dim, self.model_dim, bias=False)

        if self.apply_gated_head:
            self.neighbor = nn.Linear(model_dim, neighbor_dim, bias=bias)

            if self.concat_input:
                self.gate = nn.Linear(2 * model_dim + neighbor_dim, self.num_heads, bias=bias)
            else:
                self.gate = nn.Linear(model_dim + neighbor_dim, self.num_heads, bias=bias)

        self.sigmoid = nn.Sigmoid()
        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.value = nn.Linear(model_dim, num_heads * self.d_v, bias=False)
        self.relation = nn.Linear(self.relation_dim, num_heads * self.relation_d_v, bias=False)
        self.weight_vector_value_src = nn.Parameter(torch.Tensor(1, self.num_heads, self.d_v))
        self.weight_vector_value_tgt = nn.Parameter(torch.Tensor(1, self.num_heads, self.d_v))
        self.weight_vector_value_relation = nn.Parameter(torch.Tensor(1, self.num_heads, self.relation_d_v))
        # self.final_proj_weight = nn.Parameter(torch.Tensor(self.num_heads, 2 * self.d_v, self.d_v))
        # self.final_proj_bias = nn.Parameter(torch.Tensor(1, self.num_heads, self.d_v))

        for head_idx in range(self.num_heads):
            setattr(self, f'final_proj{head_idx}', nn.Linear(
                2 * self.d_v,
                self.d_v,
                bias=True
            ))

        if self.is_final_bias:
            self.final_bias = nn.Parameter(torch.Tensor(self.model_dim))
        else:
            self.final_bias = None

        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.relation.weight)
        # nn.init.xavier_uniform_(self.final_proj_weight)
        nn.init.xavier_uniform_(self.weight_vector_value_src)
        nn.init.xavier_uniform_(self.weight_vector_value_tgt)
        nn.init.xavier_uniform_(self.weight_vector_value_relation)

        for head_idx in range(self.num_heads):
            final_proj = getattr(self, f'final_proj{head_idx}')
            nn.init.xavier_uniform_(final_proj.weight)
            nn.init.zeros_(final_proj.bias)

        if self.is_final_bias:
            nn.init.zeros_(self.final_bias)

        if self.apply_gated_head:
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.xavier_uniform_(self.neighbor.weight)

        if self.proj_self_edge:
            nn.init.xavier_uniform_(self.self_weight.weight)

        # nn.init.zeros_(self.final_proj_bias)

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

    def calculate_output_from_scores(
        self,
        scores,
        tgt_index,
        num_nodes,
        dist_edge,
        node_src,
        value
    ):
        # 4) Apply softmax over neighborhood to produce attention.
        scaled_scores = scores - scores.max()
        exp_scores = scaled_scores.exp()

        tgt_index_broadcasted = self.explicit_broadcast(tgt_index, exp_scores)
        # node_size x nhead
        neighbor_sum = torch.zeros((num_nodes, self.num_heads)).to(exp_scores)
        neighbor_sum.scatter_add_(0, tgt_index_broadcasted, exp_scores)

        # edge_size x nhead
        neighbor_aware_denominator = neighbor_sum.index_select(0, tgt_index)
        attn = exp_scores / (neighbor_aware_denominator + (1 / word.INFINITY_NUMBER))

        if self.normalized:
            # edge_size x 1
            unsq_dist_edge = dist_edge.unsqueeze(-1)
            # edge_size x nhead
            attn = attn / unsq_dist_edge
            neighbor_attn_sum = torch.zeros((num_nodes, self.num_heads)).to(attn)
            tgt_index_broadcasted = self.explicit_broadcast(tgt_index, attn)
            neighbor_attn_sum.scatter_add_(0, tgt_index_broadcasted, attn)
            attn = attn / neighbor_attn_sum.index_select(0, tgt_index)

        # edge_size x nhead x 1
        attn = attn.unsqueeze(-1)
        attn = self.dropout(attn)

        # 5) Update final node representation.
        # edge_size x nhead x d_v
        weighted_node_src = node_src * attn
        tgt_index_broadcasted = self.explicit_broadcast(tgt_index, weighted_node_src)
        # node_size x nhead x d_v
        output = torch.zeros((num_nodes, self.num_heads, self.d_v)).to(value)
        output.scatter_add_(0, tgt_index_broadcasted, weighted_node_src)

        return output, attn

    def forward(
        self,
        inp,
        relation_inp,
        dep_rel_pos_edge,
        word_rel_pos_edge,
        deprel_edge,
        deparc_edge,
        deprel_ext_edge,
        edge_index,
        dist_edge,
        deprel_path_edge,
        deparc_path_edge,
        path_len_edge,
        deprel_ext_path_edge
    ):
        num_nodes = inp.size(0)

        # 1) Project value.
        # node_size x nhead x d_v
        value = self.value(inp).view(-1, self.num_heads, self.d_v)

        acc_relation_inp = []

        if self.max_relative_position > 0:
            # rel_pos_edge: edge_size
            # Shift values to be >= 0
            if self.use_dep_rel_pos:
                if self.use_neg_dist:
                    dep_relative_positions_matrix = dep_rel_pos_edge + self.max_relative_position
                else:
                    dep_relative_positions_matrix = torch.abs(dep_rel_pos_edge)

                #  edge_size x model_dim
                acc_relation_inp.append(self.dep_relative_positions_embeddings(
                    dep_relative_positions_matrix.to(value.device)
                ))
            if self.use_word_rel_pos:
                if self.use_neg_dist:
                    word_relative_positions_matrix = word_rel_pos_edge + self.max_relative_position
                else:
                    word_relative_positions_matrix = torch.abs(word_rel_pos_edge)

                #  edge_size x model_dim
                acc_relation_inp.append(self.word_relative_positions_embeddings(
                    word_relative_positions_matrix.to(value.device)
                ))
        if self.use_deprel:
            relations_deprels = self.deprel_embeddings(
                deprel_edge.to(value.device)
            )
            relations_deparcs = self.deparc_embeddings(
                deparc_edge.to(value.device)
            )

            #  edge_size x model_dim
            acc_relation_inp.append(torch.cat((relations_deprels, relations_deparcs), -1))
        elif self.use_deprel_ext:
            acc_relation_inp.append(self.deprel_ext_embeddings(
                deprel_ext_edge.to(value.device)
            ))
        elif self.use_dep_path or self.use_dep_ext_path:
            relations = self.dep_path_embeddings(
                deprel_path_edge=deprel_path_edge,
                deparc_path_edge=deparc_path_edge,
                path_len_edge=path_len_edge,
                deprel_ext_path_edge=deprel_ext_path_edge
            )

            acc_relation_inp.append(relations)

        relation_inp = torch.cat(acc_relation_inp, dim=-1)

        if self.omit_self_edge:
            filtered_idx = deparc_edge != self.self_idx
            f_relation_inp = relation_inp[filtered_idx]
            f_edge_index = edge_index[:, filtered_idx]
            f_dist_edge = dist_edge[filtered_idx]
        else:
            f_relation_inp = relation_inp
            f_edge_index = edge_index
            f_dist_edge = dist_edge

        f_num_edges = f_edge_index.size(1)

        # 2) Project relation.
        # edge_size x nhead x relation_d_v
        relation = self.relation(f_relation_inp).view(-1, self.num_heads, self.relation_d_v)

        # 3) Concatenate source and target node representations, multiply with weight vector, and apply leaky ReLU.
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
        scores_value = self.leakyReLU(scores_src + scores_tgt)
        # edge_size x nhead
        scores_relation = (relation * self.weight_vector_value_relation).sum(dim=-1)
        scores_relation = self.leakyReLU(scores_relation)

        output_value, attn_value = self.calculate_output_from_scores(
            scores=scores_value,
            tgt_index=tgt_index,
            num_nodes=num_nodes,
            dist_edge=f_dist_edge,
            node_src=node_src,
            value=value
        )

        output_relation, attn_relation = self.calculate_output_from_scores(
            scores=scores_relation,
            tgt_index=tgt_index,
            num_nodes=num_nodes,
            dist_edge=f_dist_edge,
            node_src=node_src,
            value=value
        )

        # node_size x num_heads x (2 * d_v)
        out = torch.cat((output_value, output_relation), dim=-1)

        # num_heads x node_size x (2 * d_v)
        out = out.transpose(0, 1)

        res = []

        for head_idx in range(self.num_heads):
            res.append(getattr(self, f'final_proj{head_idx}')(out[head_idx]))

        output = torch.stack(res).to(out.device)
        # node_size x num_heads x d_v
        output = output.transpose(0, 1).contiguous()

        # output = torch.einsum('nij,ijk->nik', out, self.final_proj_weight) + self.final_proj_bias
        # output = output.contiguous()

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
                gate = torch.cat((inp, max_gate, avg_gate), dim=-1)
            else:
                gate = torch.cat((max_gate, avg_gate), dim=-1)

            # node_dim x nhead x 1
            gate = self.sigmoid(self.gate(gate)).unsqueeze(-1)
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
        attn_value_per_head = [attn.squeeze(1) for attn in attn_value.squeeze(-1).chunk(self.num_heads, dim=1)]
        attn_relation_per_head = [attn.squeeze(1) for attn in attn_relation.squeeze(-1).chunk(self.num_heads, dim=1)]

        return output, attn_value_per_head, attn_relation_per_head
