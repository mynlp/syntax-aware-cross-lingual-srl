import math
import torch
import torch.nn as nn
from constants import word
from utils import model
from models.positional_embedding import PositionalEmbedding
from models.edge_dependency_path_encoder import EdgeDependencyPathEncoder


class SAN(nn.Module):
    def __init__(
        self,
        deparc_voc,
        num_heads,
        model_dim,
        neighbor_dim,
        d_k,
        d_v,
        dropout,
        normalized,
        bias,
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
        omit_self_edge,
        proj_self_edge,
        concat_input,
        apply_gated_head,
        use_positional_embedding,
        use_dep_rel_pos,
        use_word_rel_pos,
        lstm_dropout,
        sum_dep_path
    ):
        super(SAN, self).__init__()

        self.num_heads = num_heads
        self.model_dim = model_dim
        self.d_k = d_k
        self.d_v = d_v
        self.normalized = normalized
        self.neighbor_dim = neighbor_dim
        self.max_relative_position = max_relative_position
        self.use_neg_dist = use_neg_dist
        self.use_deprel = use_deprel
        self.use_deprel_ext = use_deprel_ext
        self.use_dep_path = use_dep_path
        self.use_dep_ext_path = use_dep_ext_path
        self.omit_self_edge = omit_self_edge
        self.proj_self_edge = proj_self_edge
        self.concat_input = concat_input
        self.is_bias = bias
        self.apply_gated_head = apply_gated_head
        self.use_positional_embedding = use_positional_embedding
        self.use_dep_rel_pos = use_dep_rel_pos
        self.use_word_rel_pos = use_word_rel_pos

        self.self_idx = deparc_voc[word.deparc_map['self']]

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        self.key = nn.Linear(model_dim, num_heads * self.d_k, bias=bias)
        self.query = nn.Linear(model_dim, num_heads * self.d_k, bias=bias)
        self.value = nn.Linear(model_dim, num_heads * self.d_v, bias=bias)

        if self.concat_input:
            self.output = nn.Linear((model_dim + num_heads * self.d_v), model_dim, bias=bias)
        else:
            self.output = nn.Linear(num_heads * self.d_v, model_dim, bias=bias)

        if self.apply_gated_head:
            self.neighbor = nn.Linear(model_dim, neighbor_dim, bias=bias)
            if self.concat_input:
                self.gate = nn.Linear(2 * model_dim + neighbor_dim, self.num_heads, bias=bias)
            else:
                self.gate = nn.Linear(model_dim + neighbor_dim, self.num_heads, bias=bias)

        if self.proj_self_edge:
            self.self_weight = nn.Linear(self.model_dim, self.model_dim, bias=False)

        self.is_relation = self.max_relative_position > 0 or \
            self.use_deprel or self.use_deprel_ext or \
            self.use_dep_path or self.use_dep_ext_path

        if self.max_relative_position > 0:
            if self.use_dep_rel_pos:
                if self.use_positional_embedding:
                    self.dep_relative_positions_embeddings_k = PositionalEmbedding(
                        self.d_k
                    )
                    self.dep_relative_positions_embeddings_v = PositionalEmbedding(
                        self.d_v
                    )
                else:
                    vocab_size = self.max_relative_position * 2 + 1 \
                        if self.use_neg_dist else self.max_relative_position + 1
                    self.dep_relative_positions_embeddings_k = nn.Embedding(
                        vocab_size,
                        self.d_k
                    )
                    self.dep_relative_positions_embeddings_v = nn.Embedding(
                        vocab_size,
                        self.d_v
                    )
            if self.use_word_rel_pos:
                if self.use_positional_embedding:
                    self.word_relative_positions_embeddings_k = PositionalEmbedding(
                        self.d_k
                    )
                    self.word_relative_positions_embeddings_v = PositionalEmbedding(
                        self.d_v
                    )
                else:
                    vocab_size = self.max_relative_position * 2 + 1 \
                        if self.use_neg_dist else self.max_relative_position + 1
                    self.word_relative_positions_embeddings_k = nn.Embedding(
                        vocab_size,
                        self.d_k
                    )
                    self.word_relative_positions_embeddings_v = nn.Embedding(
                        vocab_size,
                        self.d_v
                    )
        if self.use_deprel:
            self.deprel_embeddings_k = nn.Embedding(
                deprel_size,
                deprel_edge_dim,
                padding_idx=word.PAD
            )
            self.deparc_embeddings_k = nn.Embedding(
                deparc_size,
                deparc_edge_dim,
                padding_idx=word.PAD
            )
            self.deprel_embeddings_v = nn.Embedding(
                deprel_size,
                deprel_edge_dim,
                padding_idx=word.PAD
            )
            self.deparc_embeddings_v = nn.Embedding(
                deparc_size,
                deparc_edge_dim,
                padding_idx=word.PAD
            )
        elif self.use_deprel_ext:
            self.deprel_ext_embeddings_k = nn.Embedding(
                deprel_ext_size,
                d_k,
                padding_idx=word.PAD
            )
            self.deprel_ext_embeddings_v = nn.Embedding(
                deprel_ext_size,
                d_v,
                padding_idx=word.PAD
            )
        elif self.use_dep_path or self.use_dep_ext_path:
            self.dep_path_embeddings_k = EdgeDependencyPathEncoder(
                input_size=d_k,
                hidden_size=d_k,
                dropout=lstm_dropout,
                deprel_size=deprel_size,
                deparc_size=deparc_size,
                deprel_ext_size=deprel_ext_size,
                deprel_edge_dim=deprel_edge_dim,
                deparc_edge_dim=deparc_edge_dim,
                deprel_ext_edge_dim=d_k,
                use_dep_path=use_dep_path,
                use_dep_ext_path=use_dep_ext_path,
                sum_dep_path=sum_dep_path
            )

            self.dep_path_embeddings_v = EdgeDependencyPathEncoder(
                input_size=d_v,
                hidden_size=d_v,
                dropout=lstm_dropout,
                deprel_size=deprel_size,
                deparc_size=deparc_size,
                deprel_ext_size=deprel_ext_size,
                deprel_edge_dim=deprel_edge_dim,
                deparc_edge_dim=deparc_edge_dim,
                deprel_ext_edge_dim=d_v,
                use_dep_path=use_dep_path,
                use_dep_ext_path=use_dep_ext_path,
                sum_dep_path=sum_dep_path
            )

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.output.weight)

        if self.apply_gated_head:
            nn.init.xavier_uniform_(self.neighbor.weight)
            nn.init.xavier_uniform_(self.gate.weight)

        if self.proj_self_edge:
            nn.init.xavier_uniform_(self.self_weight.weight)

        if self.is_bias:
            nn.init.zeros_(self.key.bias)
            nn.init.zeros_(self.query.bias)
            nn.init.zeros_(self.value.bias)
            nn.init.zeros_(self.output.bias)

            if self.apply_gated_head:
                nn.init.zeros_(self.neighbor.bias)
                nn.init.zeros_(self.gate.bias)

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
        word_rel_pos_edge,
        dep_rel_pos_edge,
        edge_index,
        dist_edge,
        deprel_ext_edge,
        deprel_path_edge,
        deparc_path_edge,
        path_len_edge,
        deprel_ext_path_edge
    ):
        num_nodes = inp.size(0)

        # node_size x nhead x d_k
        key = self.key(inp).view(-1, self.num_heads, self.d_k)
        # node_size x nhead x d_k
        query = self.query(inp).view(-1, self.num_heads, self.d_k)
        # node_size x nhead x d_v
        value = self.value(inp).view(-1, self.num_heads, self.d_v)

        acc_relations_keys = []
        acc_relations_values = []

        if self.max_relative_position > 0:
            if self.use_dep_rel_pos:
                if self.use_neg_dist:
                    dep_relative_positions_matrix = dep_rel_pos_edge + self.max_relative_position
                else:
                    dep_relative_positions_matrix = torch.abs(dep_rel_pos_edge)

                # edge_size x d_k
                acc_relations_keys.append(self.dep_relative_positions_embeddings_k(
                    dep_relative_positions_matrix.to(key.device)
                ))
                # edge_size x d_v
                acc_relations_values.append(self.dep_relative_positions_embeddings_v(
                    dep_relative_positions_matrix.to(value.device)
                ))
            if self.use_word_rel_pos:
                if self.use_neg_dist:
                    word_relative_positions_matrix = word_rel_pos_edge + self.max_relative_position
                else:
                    word_relative_positions_matrix = torch.abs(word_rel_pos_edge)

                # edge_size x d_k
                acc_relations_keys.append(self.word_relative_positions_embeddings_k(
                    word_relative_positions_matrix.to(key.device)
                ))
                # edge_size x d_v
                acc_relations_values.append(self.word_relative_positions_embeddings_v(
                    word_relative_positions_matrix.to(value.device)
                ))
        if self.use_deprel:
            relations_deprel_keys = self.deprel_embeddings_k(
                deprel_edge.to(key.device)
            )
            relations_deparc_keys = self.deparc_embeddings_k(
                deparc_edge.to(key.device)
            )
            relations_deprel_values = self.deprel_embeddings_v(
                deprel_edge.to(key.device)
            )
            relations_deparc_values = self.deparc_embeddings_v(
                deparc_edge.to(key.device)
            )
            #  edge_size x d_k
            acc_relations_keys.append(torch.cat((relations_deprel_keys, relations_deparc_keys), -1))
            #  edge_size x d_v
            acc_relations_values.append(torch.cat((relations_deprel_values, relations_deparc_values), -1))
        elif self.use_deprel_ext:
            acc_relations_keys.append(self.deprel_ext_embeddings_k(
                deprel_ext_edge.to(key.device)
            ))
            acc_relations_values.append(self.deprel_ext_embeddings_v(
                deprel_ext_edge.to(key.device)
            ))
        elif (self.use_dep_path or self.use_dep_ext_path):
            relations_keys = self.dep_path_embeddings_k(
                deprel_path_edge=deprel_path_edge,
                deparc_path_edge=deparc_path_edge,
                path_len_edge=path_len_edge,
                deprel_ext_path_edge=deprel_ext_path_edge
            )

            relations_values = self.dep_path_embeddings_v(
                deprel_path_edge=deprel_path_edge,
                deparc_path_edge=deparc_path_edge,
                path_len_edge=path_len_edge,
                deprel_ext_path_edge=deprel_ext_path_edge
            )

            acc_relations_keys.append(relations_keys)
            acc_relations_values.append(relations_values)

        if self.omit_self_edge:
            filtered_idx = deparc_edge != self.self_idx

            for idx in range(len(acc_relations_keys)):
                acc_relations_keys[idx] = acc_relations_keys[idx][filtered_idx]
                acc_relations_values[idx] = acc_relations_values[idx][filtered_idx]

            f_edge_index = edge_index[:, filtered_idx]
            f_dist_edge = dist_edge[filtered_idx]
        else:
            f_edge_index = edge_index
            f_dist_edge = dist_edge

        f_num_edges = f_edge_index.size(1)
        src_index = f_edge_index[0]
        tgt_index = f_edge_index[1]

        if self.is_relation:
            for idx in range(len(acc_relations_keys)):
                # edge_size x nhead x d_k
                acc_relations_keys[idx] = acc_relations_keys[idx].unsqueeze(1).repeat(1, self.num_heads, 1)
                # edge_size x nhead x d_v
                acc_relations_values[idx] = acc_relations_values[idx].unsqueeze(1).repeat(1, self.num_heads, 1)

        # edge_size x nhead x d_k
        node_key = key.index_select(0, src_index)
        # edge_size x nhead x d_k
        node_query = query.index_select(0, tgt_index)
        # edge_size x nhead x d_v
        node_value = value.index_select(0, src_index)

        scores = (node_query * node_key).sum(dim=-1)
        # edge_size x nhead
        if self.is_relation:
            for relations_keys in acc_relations_keys:
                scores += (node_query * relations_keys).sum(dim=-1)

        scores /= math.sqrt(self.d_k)
        scores = scores - scores.max()
        exp_scores = scores.exp()

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
            tgt_index_broadcasted = self.explicit_broadcast(tgt_index, attn)
            neighbor_attn_sum.scatter_add_(0, tgt_index_broadcasted, attn)
            attn = attn / neighbor_attn_sum.index_select(0, tgt_index)

        # edge_size x nhead x 1
        attn = attn.unsqueeze(-1)
        attn = self.dropout(attn)

        weighted_node_src = node_value * attn

        if self.is_relation:
            for relations_values in acc_relations_values:
                weighted_node_src += (relations_values * attn)

        output = torch.zeros((num_nodes, self.num_heads, self.d_v)).to(value)
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

            # node_size x nhead x d_v
            output = gate * output

        output = output.view(-1, self.num_heads * self.d_v)

        if self.concat_input:
            # node_size x model_dim (=num_heads x d_v)
            output = self.output(torch.cat((inp, output), dim=-1))
        else:
            # node_size x model_dim (=num_heads x d_v)
            output = self.output(output)

        if self.proj_self_edge:
            output = output + self.self_weight(inp)

        # a list of size num_heads containing tensors
        # of shape edge_size
        attn_per_head = [att.squeeze(1) for att in attn.squeeze(-1).chunk(self.num_heads, dim=1)]

        return output, attn_per_head
