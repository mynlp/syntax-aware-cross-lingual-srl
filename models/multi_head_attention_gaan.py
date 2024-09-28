import torch
import math
import torch.nn as nn
import torch.nn.functional as f
from utils import model
from constants import word
from models.positional_embedding import PositionalEmbedding
from models.dependency_path_encoder import DependencyPathEncoder


class MultiHeadedAttentionGaan(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       num_heads (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by num_heads
       dropout (float): dropout parameter
    """

    def __init__(
        self,
        num_heads,
        model_dim,
        d_k,
        d_v,
        dropout,
        max_relative_position,
        dep_max_relative_position,
        use_neg_dist,
        use_deprel,
        use_dep_path,
        use_deprel_ext,
        use_dep_ext_path,
        sum_dep_path,
        deprel_size,
        deparc_size,
        deprel_ext_size,
        deprel_edge_dim,
        deparc_edge_dim,
        neighbor_dim,
        normalized,
        bias,
        omit_self_edge,
        proj_self_edge,
        concat_input,
        use_positional_embedding,
        use_dep_rel_pos,
        use_word_rel_pos,
        lstm_dropout
    ):
        super(MultiHeadedAttentionGaan, self).__init__()

        self.num_heads = num_heads
        self.model_dim = model_dim
        self.d_k = d_k
        self.d_v = d_v
        self.is_bias = bias
        self.omit_self_edge = omit_self_edge
        self.proj_self_edge = proj_self_edge
        self.concat_input = concat_input
        self.use_positional_embedding = use_positional_embedding
        self.use_dep_rel_pos = use_dep_rel_pos
        self.use_word_rel_pos = use_word_rel_pos

        self.key = nn.Linear(model_dim, num_heads * self.d_k, bias=bias)
        self.query = nn.Linear(model_dim, num_heads * self.d_k, bias=bias)
        self.value = nn.Linear(model_dim, num_heads * self.d_v, bias=bias)
        self.neighbor = nn.Linear(model_dim, neighbor_dim, bias=bias)

        if self.proj_self_edge:
            self.self_weight = nn.Linear(model_dim, model_dim, bias=False)

        if self.concat_input:
            self.gate = nn.Linear(2 * model_dim + neighbor_dim, self.num_heads, bias=bias)
            self.output = nn.Linear((model_dim + self.num_heads * d_v), model_dim, bias=bias)
        else:
            self.gate = nn.Linear(model_dim + neighbor_dim, self.num_heads, bias=bias)
            self.output = nn.Linear(self.num_heads * d_v, model_dim, bias=bias)

        self.leakyReLU = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.max_relative_position = max_relative_position
        self.dep_max_relative_position = dep_max_relative_position
        self.use_neg_dist = use_neg_dist
        self.use_deprel = use_deprel
        self.use_deprel_ext = use_deprel_ext
        self.use_dep_path = use_dep_path
        self.use_dep_ext_path = use_dep_ext_path
        self.normalized = normalized
        self.is_relation = self.max_relative_position > 0 or \
            self.use_deprel or self.use_deprel_ext or \
            self.use_dep_path or self.use_dep_ext_path

        if self.use_word_rel_pos:
            assert self.max_relative_position > 0
            if self.use_positional_embedding:
                self.word_relative_positions_embeddings_k = PositionalEmbedding(
                    self.d_k
                )
                self.word_relative_positions_embeddings_v = PositionalEmbedding(
                    self.d_v
                )
            else:
                vocab_size = word.DUMMY_MAX_LEN * 2 + 1 \
                    if self.use_neg_dist else word.DUMMY_MAX_LEN + 1
                self.word_relative_positions_embeddings_k = nn.Embedding(
                    vocab_size,
                    self.d_k,
                    padding_idx=word.DUMMY_MAX_LEN
                )
                self.word_relative_positions_embeddings_v = nn.Embedding(
                    vocab_size,
                    self.d_v,
                    padding_idx=word.DUMMY_MAX_LEN
                )
        if self.use_dep_rel_pos:
            assert self.dep_max_relative_position > 0
            if self.use_positional_embedding:
                self.dep_relative_positions_embeddings_k = PositionalEmbedding(
                    self.d_k
                )
                self.dep_relative_positions_embeddings_v = PositionalEmbedding(
                    self.d_v
                )
            else:
                vocab_size = word.DUMMY_MAX_LEN * 2 + 1 \
                    if self.use_neg_dist else word.DUMMY_MAX_LEN + 1
                self.dep_relative_positions_embeddings_k = nn.Embedding(
                    vocab_size,
                    self.d_k,
                    padding_idx=word.DUMMY_MAX_LEN
                )
                self.dep_relative_positions_embeddings_v = nn.Embedding(
                    vocab_size,
                    self.d_v,
                    padding_idx=word.DUMMY_MAX_LEN
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
        elif (self.use_dep_path or self.use_dep_ext_path):
            self.dep_path_embeddings_k = DependencyPathEncoder(
                input_size=d_k,
                hidden_size=d_k,
                dropout=lstm_dropout,
                deprel_size=deprel_size,
                deparc_size=deparc_size,
                deprel_ext_size=deprel_ext_size,
                deprel_edge_dim=deprel_edge_dim,
                deparc_edge_dim=deparc_edge_dim,
                use_dep_ext_path=use_dep_ext_path,
                use_dep_path=use_dep_path,
                sum_dep_path=sum_dep_path
            )

            self.dep_path_embeddings_v = DependencyPathEncoder(
                input_size=d_v,
                hidden_size=d_v,
                dropout=lstm_dropout,
                deprel_size=deprel_size,
                deparc_size=deparc_size,
                deprel_ext_size=deprel_ext_size,
                deprel_edge_dim=deprel_edge_dim,
                deparc_edge_dim=deparc_edge_dim,
                use_dep_ext_path=use_dep_ext_path,
                use_dep_path=use_dep_path,
                sum_dep_path=sum_dep_path
            )

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.xavier_uniform_(self.neighbor.weight)
        nn.init.xavier_uniform_(self.gate.weight)

        if self.proj_self_edge:
            nn.init.xavier_uniform_(self.self_weight.weight)

        if self.is_bias:
            nn.init.zeros_(self.key.bias)
            nn.init.zeros_(self.query.bias)
            nn.init.zeros_(self.value.bias)
            nn.init.zeros_(self.neighbor.bias)
            nn.init.zeros_(self.gate.bias)
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        key,
        value,
        query,
        mask,
        adj_mask,
        dep_rel_pos_mat,
        word_rel_pos_mat,
        deprel_mat,
        deparc_mat,
        deprel_ext_mat,
        deprel_path_mat,
        deparc_path_mat,
        path_len_mat,
        deprel_ext_path_mat
    ):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):
           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        batch_size = key.size(0)
        key_len = key.size(1)

        num_heads = self.num_heads
        original_value = value

        # 1) Project key, value, and query.
        key = model.shape(
            x=self.key(key),
            dim=self.d_k,
            batch_size=batch_size,
            num_heads=num_heads
        )  # bsz x nhead x key_len x d_k
        value = model.shape(
            x=self.leakyReLU(self.value(value)),
            dim=self.d_v,
            batch_size=batch_size,
            num_heads=num_heads
        )  # bsz x nhead x key_len x d_v
        query = model.shape(
            x=self.query(query),
            dim=self.d_k,
            batch_size=batch_size,
            num_heads=num_heads
        )  # bsz x nhead x query_len x d_k

        acc_relations_keys = []
        acc_relations_values = []

        # dep_rel_pos_mat: bsz x key_len x key_len
        # Shift values to be >= 0
        if self.use_dep_rel_pos:
            if self.use_neg_dist:
                dep_relative_positions_matrix = dep_rel_pos_mat + self.dep_max_relative_position
            else:
                dep_relative_positions_matrix = torch.abs(dep_rel_pos_mat)

            #  bsz x key_len x key_len x d_k
            acc_relations_keys.append(self.dep_relative_positions_embeddings_k(
                dep_relative_positions_matrix.to(key.device)
            ))
            #  bsz x key_len x key_len x d_v
            acc_relations_values.append(self.dep_relative_positions_embeddings_v(
                dep_relative_positions_matrix.to(key.device)
            ))
        if self.use_word_rel_pos:
            if self.use_neg_dist:
                word_relative_positions_matrix = word_rel_pos_mat + self.max_relative_position
            else:
                word_relative_positions_matrix = torch.abs(word_rel_pos_mat)

            #  bsz x key_len x key_len x d_k
            acc_relations_keys.append(self.word_relative_positions_embeddings_k(
                word_relative_positions_matrix.to(key.device)
            ))
            #  bsz x key_len x key_len x d_v
            acc_relations_values.append(self.word_relative_positions_embeddings_v(
                word_relative_positions_matrix.to(key.device)
            ))
        if self.use_deprel:
            relations_deprel_keys = self.deprel_embeddings_k(
                deprel_mat.to(key.device)
            )
            relations_deparc_keys = self.deparc_embeddings_k(
                deparc_mat.to(key.device)
            )
            relations_deprel_values = self.deprel_embeddings_v(
                deprel_mat.to(key.device)
            )
            relations_deparc_values = self.deparc_embeddings_v(
                deparc_mat.to(key.device)
            )
            #  bsz x key_len x key_len x d_k
            acc_relations_keys.append(torch.cat((relations_deprel_keys, relations_deparc_keys), -1))
            #  bsz x key_len x key_len x d_v
            acc_relations_values.append(torch.cat((relations_deprel_values, relations_deparc_values), -1))
        elif self.use_deprel_ext:
            acc_relations_keys.append(self.deprel_ext_embeddings_k(
                deprel_ext_mat.to(key.device)
            ))
            acc_relations_values.append(self.deprel_ext_embeddings_v(
                deprel_ext_mat.to(key.device)
            ))
        elif (self.use_dep_path or self.use_dep_ext_path):
            relations_keys = self.dep_path_embeddings_k(
                deprel_path_mat=deprel_path_mat,
                deparc_path_mat=deparc_path_mat,
                path_len_mat=path_len_mat,
                deprel_ext_path_mat=deprel_ext_path_mat
            )

            relations_values = self.dep_path_embeddings_v(
                deprel_path_mat=deprel_path_mat,
                deparc_path_mat=deparc_path_mat,
                path_len_mat=path_len_mat,
                deprel_ext_path_mat=deprel_ext_path_mat
            )

            acc_relations_keys.append(relations_keys)
            acc_relations_values.append(relations_values)

        # 2) Calculate and scale scores.
        # bsz x nhead x query_len x d_k
        query = query / math.sqrt(self.d_k)
        # batch x nhead x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))
        scores = query_key

        if self.is_relation:
            # query_len x bsz x nhead x d_k
            permuted_query = query.permute(2, 0, 1, 3)

            for relations_keys in acc_relations_keys:
                assert relations_keys.dim() == 4
                # permuted_relations_keys: key_len x bsz x d_k x key_len
                permuted_relations_keys = relations_keys.permute(1, 0, 3, 2)
                # scores: key_len x bsz x nhead x key_len
                cur_scores = torch.matmul(permuted_query, permuted_relations_keys)
                # scores: bsz x nhead x query_len x key_len
                scores += cur_scores.permute(1, 2, 0, 3)

        # we attend to every element in the key/value for a query
        scores = scores.float()  # bsz x nhead x query_len x key_len

        if self.omit_self_edge:
            mask = mask | torch.eye(key_len).repeat(batch_size, 1, 1).bool().to(mask)

        mask = mask.unsqueeze(1)  # [B, 1, query_len, key_len]
        scores = scores.masked_fill(mask.bool(), -word.INFINITY_NUMBER)

        if adj_mask is not None:
            assert adj_mask.size() == scores.size()
            scores = scores.masked_fill(~adj_mask.bool(), -word.INFINITY_NUMBER)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        if adj_mask is not None and self.normalized:
            assert adj_mask.size() == attn.size()
            adj_mask_ = adj_mask.masked_fill(~adj_mask.bool(), word.INFINITY_NUMBER)
            adj_mask_ = 1.0 / adj_mask_
            attn = attn * adj_mask_
            attn = f.normalize(attn, p=1, dim=-1)

        # TODO: Check if this is true.
        # To connect only desirable edges.
        attn = attn.masked_fill(mask.bool(), 0)
        if adj_mask is not None:
            attn = attn.masked_fill(~adj_mask.bool(), 0)

        # bsz x nhead x query_len x key_len
        attn = self.dropout(attn)
        # bsz x nhead x query_len x d_v
        context_original = torch.matmul(attn, value)

        if self.is_relation:
            # permuted_attn: query_len x bsz x nhead x key_len
            permuted_attn = attn.permute(2, 0, 1, 3)

            for relations_values in acc_relations_values:
                assert relations_values.dim() == 4

                # relations_values: key_len x bsz x key_len x d_v
                add_term = torch.matmul(
                    permuted_attn,
                    relations_values.transpose(0, 1)
                )
                # add_term: key_len x bsz x nhead x d_v
                add_term = add_term.permute(1, 2, 0, 3)
                context_original += add_term

        context = context_original.transpose(1, 2).contiguous()  # bsz x query_len x nhead x d_v

        # bsz x key_len x neighbor_dim
        neighbor = self.neighbor(original_value)

        max_gate, avg_gate = model.apply_gate_to_weight_heads_in_trans(
            neighbor=neighbor,
            key_len=key_len,
            mask=mask,
            adj_mask=adj_mask,
            original_value=original_value
        )

        if self.concat_input:
            # bsz x key_len x nhead
            gate = self.sigmoid(self.gate(torch.cat((original_value, max_gate, avg_gate), dim=-1)))
            # bsz x key_len x nhead x 1
            gate = gate.unsqueeze(-1)
            # bsz x key_len x d_model
            context = (gate * context).view(batch_size, -1, num_heads * self.d_v)
            final_output = self.output(torch.cat((original_value, context), dim=-1))  # bsz x query_len x d_model
        else:
            # bsz x key_len x nhead
            gate = self.sigmoid(self.gate(torch.cat((max_gate, avg_gate), dim=-1)))
            # bsz x key_len x nhead x 1
            gate = gate.unsqueeze(-1)
            # bsz x key_len x d_model
            context = (gate * context).view(batch_size, -1, num_heads * self.d_v)
            final_output = self.output(context)  # bsz x query_len x d_model

        if self.proj_self_edge:
            final_output += self.self_weight(original_value)

        # a list of size num_heads containing tensors
        # of shape `batch x query_len x key_len`
        attn_per_head = [attn.squeeze(1) for attn in attn.chunk(num_heads, dim=1)]

        return final_output, attn_per_head
