import torch
import torch.nn as nn
import torch.nn.functional as f
from utils import model
from constants import word


class TwoAttentionGAT(nn.Module):
    def __init__(
        self,
        num_heads,
        model_dim,
        dropout,
        max_relative_position,
        use_neg_dist,
        use_deprel,
        use_dep_path,
        deprel_size,
        deparc_size,
        deprel_edge_dim,
        deparc_edge_dim,
        normalized,
        concat,
        activation,
        dropout_init,
        dropout_final,
        bias,
        init_weight_xavier_uniform
    ):
        super(TwoAttentionGAT, self).__init__()

        self.is_dropout_init = dropout_init
        self.is_dropout_final = dropout_final
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.is_concat = concat
        self.activation = activation
        self.is_init_weight_xavier_uniform = init_weight_xavier_uniform

        assert self.model_dim % self.num_heads == 0
        if self.is_concat:
            self.d_v = int(self.model_dim / self.num_heads)
        else:
            self.d_v = self.model_dim

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.value = nn.Linear(model_dim, num_heads * self.d_v, bias=bias)
        self.relation = nn.Linear(model_dim, num_heads * self.d_v, bias=bias)
        self.weight_vector_value_src = nn.Parameter(torch.Tensor(1, 1, 1, self.num_heads, self.d_v))
        self.weight_vector_value_tgt = nn.Parameter(torch.Tensor(1, 1, 1, self.num_heads, self.d_v))
        self.weight_vector_value_relation = nn.Parameter(torch.Tensor(1, 1, 1, self.num_heads, self.d_v))
        self.final_proj = nn.Linear(model_dim * 2, model_dim, bias=True)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.max_relative_position = max_relative_position
        self.use_neg_dist = use_neg_dist
        self.use_deprel = use_deprel
        self.use_dep_path = use_dep_path
        self.normalized = normalized

        if max_relative_position > 0:
            vocab_size = word.DUMMY_MAX_LEN * 2 + 1 \
                if self.use_neg_dist else word.DUMMY_MAX_LEN + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size,
                self.model_dim,
                padding_idx=word.DUMMY_MAX_LEN
            )
        elif self.use_deprel:
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

        self.init_params()

    def init_params(self):
        if self.is_init_weight_xavier_uniform:
            nn.init.xavier_uniform_(self.value.weight)
            nn.init.xavier_uniform_(self.relation.weight)
            nn.init.xavier_uniform_(self.final_proj.weight)

        nn.init.xavier_uniform_(self.weight_vector_value_src)
        nn.init.xavier_uniform_(self.weight_vector_value_tgt)
        nn.init.xavier_uniform_(self.weight_vector_value_relation)

    def forward(
        self,
        inp,
        relation_inp,
        mask,
        adj_mask,
        rel_pos_mat,
        deprel_mat,
        deparc_mat
    ):
        batch_size = inp.size(0)
        num_heads = self.num_heads

        if self.is_dropout_init:
            do_inp = self.dropout(inp)
        else:
            do_inp = inp

        # 1) Project value.
        value = model.shape(
            x=self.value(do_inp),
            dim=self.d_v,
            batch_size=batch_size,
            num_heads=num_heads
        )  # bsz x nhead x value_len x d_v

        if self.is_dropout_init:
            value = self.dropout(value)

        if self.max_relative_position > 0:
            # dep_rel_pos_mat: bsz x value_len x value_len
            # Shift values to be >= 0
            if self.use_neg_dist:
                relative_positions_matrix = rel_pos_mat + self.max_relative_position
            else:
                relative_positions_matrix = torch.abs(rel_pos_mat)

            #  bsz x value_len x value_len x d_v
            relation_inp = self.relative_positions_embeddings(
                relative_positions_matrix.to(value.device)
            )
        elif self.use_deprel:
            relations_deprels = self.deprel_embeddings(
                deprel_mat.to(value.device)
            )
            relations_deparcs = self.deparc_embeddings(
                deparc_mat.to(value.device)
            )

            #  bsz x value_len x value_len x d_v
            relation_inp = torch.cat((relations_deprels, relations_deparcs), -1)
        else:
            assert relation_inp is not None

        # 2) Project relation.
        _, value_len, _, _ = relation_inp.size()

        if self.is_dropout_init:
            relation_inp = self.dropout(relation_inp)

        relation = self.relation(relation_inp).view(
            batch_size,
            value_len,
            value_len,
            num_heads,
            self.d_v
        )  # bsz x value_len x value_len x nhead x d_v

        if self.is_dropout_init:
            relation = self.dropout(relation)

        # 3) Concatenate source and target node representations, multiply with weight vector, and apply leaky ReLU.
        _, _, value_len, _ = value.size()  # bsz x nhead x value_len x d_v
        # bsz x nhead x value_len x value_len x d_v
        value_src = value.unsqueeze(2).repeat(1, 1, value_len, 1, 1)
        # bsz x nhead x value_len x value_len x d_v
        value_tgt = value.unsqueeze(3).repeat(1, 1, 1, value_len, 1)

        # bsz x value_len x value_len x nhead x d_v
        value_src = value_src.permute(0, 2, 3, 1, 4)
        # bsz x value_len x value_len x nhead x d_v
        value_tgt = value_tgt.permute(0, 2, 3, 1, 4)

        # bsz x value_len x value_len x nhead
        scores_src = (value_src * self.weight_vector_value_src).sum(dim=-1)
        # bsz x value_len x value_len x nhead
        scores_tgt = (value_tgt * self.weight_vector_value_tgt).sum(dim=-1)
        # bsz x nhead x value_len x value_len
        scores_value = (scores_src + scores_tgt).permute(0, 3, 1, 2)
        scores_value = self.leakyReLU(scores_value)

        # 4) Multiply relation with weight vector, and apply leaky ReLU.
        scores_relation = (relation * self.weight_vector_value_relation).sum(dim=-1)
        # bsz x nhead x value_len x value_len
        scores_relation = scores_relation.permute(0, 3, 1, 2)
        scores_relation = self.leakyReLU(scores_relation)

        # 5) Apply mask for padding.
        mask = mask.unsqueeze(1)  # [B, 1, query_len, key_len]
        scores_value = scores_value.masked_fill(mask, -word.INFINITY_NUMBER)
        scores_relation = scores_relation.masked_fill(mask, -word.INFINITY_NUMBER)

        if adj_mask is not None:
            assert adj_mask.size() == scores_value.size()
            assert adj_mask.size() == scores_relation.size()
            scores_value = scores_value.masked_fill(~adj_mask.bool(), -word.INFINITY_NUMBER)
            scores_relation = scores_relation.masked_fill(~adj_mask.bool(), -word.INFINITY_NUMBER)

        # 6) Apply softmax over neighborhood and normalize.
        attn_value = self.softmax(scores_value).to(value.dtype)
        attn_relation = self.softmax(scores_relation).to(relation.dtype)

        if adj_mask is not None and self.normalized:
            assert adj_mask.size() == attn_value.size()
            assert adj_mask.size() == attn_relation.size()

            adj_mask_ = adj_mask.masked_fill(~adj_mask.bool(), word.INFINITY_NUMBER)
            adj_mask_ = 1.0 / adj_mask_

            attn_value = attn_value * adj_mask_
            attn_value = f.normalize(attn_value, p=1, dim=-1)

            attn_relation = attn_relation * adj_mask_
            attn_relation = f.normalize(attn_relation, p=1, dim=-1)

        # TODO: Check if this is true.
        # To connect only desirable edges.
        attn_value = attn_value.masked_fill(mask, 0)
        attn_relation = attn_relation.masked_fill(mask, 0)
        if adj_mask is not None:
            attn_value = attn_value.masked_fill(~adj_mask.bool(), 0)
            attn_relation = attn_relation.masked_fill(~adj_mask.bool(), 0)

        # bsz x nhead x value_len x value_len
        attn_value = self.dropout(attn_value)
        attn_relation = self.dropout(attn_relation)

        # bsz x nhead x value_len x d_v
        output_original_value = torch.matmul(attn_value, value)
        output_original_relation = torch.matmul(attn_relation, value)

        if self.is_concat:
            # bsz x value_len x model_dim (=num_heads x d_v)
            output_value = model.unshape(
                x=output_original_value,
                dim=self.d_v,
                batch_size=batch_size,
                num_heads=num_heads
            )

            output_relation = model.unshape(
                x=output_original_relation,
                dim=self.d_v,
                batch_size=batch_size,
                num_heads=num_heads
            )
        else:
            # bsz x value_len x model_dim (=d_v)
            output_value = output_original_value.mean(dim=1)
            output_relation = output_original_relation.mean(dim=1)

        # bsz x value_len x model_dim * 2
        output_cat = torch.cat((output_value, output_relation), dim=-1)
        # bsz x value_len x model_dim
        final_output = self.final_proj(output_cat)

        if self.is_dropout_final:
            final_output = self.dropout(final_output)

        # Skip connection
        final_output = final_output + inp

        if self.activation is not None:
            final_output = self.activation(final_output)

        # a list of size num_heads containing tensors
        # of shape `batch x query_len x key_len`
        attn_value_per_head = [attn.squeeze(1) for attn in attn_value.chunk(num_heads, dim=1)]
        attn_relation_per_head = [attn.squeeze(1) for attn in attn_relation.chunk(num_heads, dim=1)]

        return final_output, attn_value_per_head, attn_relation_per_head
