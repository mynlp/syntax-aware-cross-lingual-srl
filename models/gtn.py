import torch
import torch.nn as nn
from models.embedder import Embedder
from models.encoder import Encoder
from utils.model import is_trans, pool, reshape_according_to_sent_len
from constants import word, model


class GTN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.embedding = Embedder(opt)

        self.transform = None
        if opt.emb_dim != opt.hid_dim:
            # to match with Transformer Encoder size, we apply transform
            self.transform = nn.Linear(opt.emb_dim, opt.hid_dim)

        self.is_trans = is_trans(opt.network_type)
        self.pred_ind_yes_idx = opt.pred_ind_yes_idx
        self.num_embed_graph_heads = opt.num_embed_graph_heads
        self.max_tree_dists = opt.max_tree_dists
        self.num_heads = opt.num_heads
        self.use_dep_rel_pos = opt.use_dep_rel_pos
        self.use_word_rel_pos = opt.use_word_rel_pos
        self.use_deprel = opt.use_deprel
        self.use_deprel_ext = opt.use_deprel_ext
        self.use_dep_path = opt.use_dep_path
        self.use_dep_ext_path = opt.use_dep_ext_path
        self.use_dep_path_from_pred = opt.use_dep_path_from_pred
        if self.num_embed_graph_heads > 0:
            assert len(self.max_tree_dists) == self.num_embed_graph_heads
            assert self.num_embed_graph_heads <= self.num_heads

        self.encoder = Encoder(opt)

        if opt.network_type[-1] == model.network_type['lstm']:
            opt.out_dim = opt.lstm_hidden_size * 2  # bidirectional
        else:
            opt.out_dim = opt.hid_dim

        self.use_sent_rep = opt.use_sent_rep
        in_dim = opt.out_dim * 3 if self.use_sent_rep else opt.out_dim * 2
        # output mlp layers
        layers = [nn.Linear(in_dim, opt.out_dim), nn.ReLU()]
        for _ in range(opt.num_mlp_layers - 1):
            layers += [nn.Linear(opt.out_dim, opt.out_dim), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def forward(self, **kwargs):
        embs = self.embedding(**kwargs)
        if self.transform is not None:
            embs = self.transform(embs)

        fc_mask_rep = kwargs.get('fc_mask_rep')
        sent_len_rep = kwargs.get('sent_len_rep')

        if self.is_trans:
            if self.num_embed_graph_heads > 0:
                adj_mask_rep = kwargs.get('adj_mask_rep')
                adj_mask_list = []
                for k in range(self.num_embed_graph_heads):
                    mask_k = torch.empty_like(adj_mask_rep).copy_(adj_mask_rep)
                    mask_k[mask_k > self.max_tree_dists[k]] = 0
                    adj_mask_list.append(mask_k.to(embs))

                no_mask_count = self.num_heads - self.num_embed_graph_heads
                for k in range(no_mask_count):
                    no_mask = torch.empty_like(fc_mask_rep).copy_(fc_mask_rep)
                    adj_mask_list.append(no_mask)

                assert len(adj_mask_list) == self.num_heads
                # B, num_head, max_len, max_len
                adj_mask_rep = torch.stack(adj_mask_list, dim=1)
            elif self.use_dep_path_from_pred:
                adj_mask_rep = kwargs.get('adj_mask_rep')
                pred_adj_mat_rep = kwargs.get('pred_adj_mat_rep')
                adj_mask_rep = adj_mask_rep.masked_fill(~pred_adj_mat_rep.bool(), 0)
                adj_mask_rep = adj_mask_rep.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            else:
                adj_mask_rep = None

            dep_rel_pos_mat_rep = kwargs.get('dep_rel_pos_mat_rep') if self.use_dep_rel_pos \
                else None
            word_rel_pos_mat_rep = kwargs.get('word_rel_pos_mat_rep') if self.use_word_rel_pos \
                else None
            deprel_mat_rep = kwargs.get('deprel_mat_rep') if self.use_deprel \
                else None
            deparc_mat_rep = kwargs.get('deparc_mat_rep')
            deprel_ext_mat_rep = kwargs.get('deprel_ext_mat_rep') if self.use_deprel_ext \
                else None
            path_len_mat_rep = kwargs.get('path_len_mat_rep') if (self.use_dep_path or self.use_dep_ext_path) \
                else None
            deprel_path_mat_rep = kwargs.get('deprel_path_mat_rep') if self.use_dep_path \
                else None
            deparc_path_mat_rep = kwargs.get('deparc_path_mat_rep') if self.use_dep_path \
                else None
            deprel_ext_path_mat_rep = kwargs.get('deprel_ext_path_mat_rep') if self.use_dep_ext_path \
                else None

            output, layer_outputs = self.encoder(
                inp=embs,
                sent_len_rep=sent_len_rep,
                adj_mask_rep=adj_mask_rep,
                dep_rel_pos_mat_rep=dep_rel_pos_mat_rep,
                word_rel_pos_mat_rep=word_rel_pos_mat_rep,
                fc_mask_rep=fc_mask_rep.bool(),
                deprel_mat_rep=deprel_mat_rep,
                deparc_mat_rep=deparc_mat_rep,
                path_len_mat_rep=path_len_mat_rep,
                deprel_path_mat_rep=deprel_path_mat_rep,
                deparc_path_mat_rep=deparc_path_mat_rep,
                deprel_ext_mat_rep=deprel_ext_mat_rep,
                deprel_ext_path_mat_rep=deprel_ext_path_mat_rep
            )
        else:
            dist_edge_rep = kwargs.get('dist_edge_rep')
            edge_len_rep = kwargs.get('edge_len_rep')
            edge_index_rep = kwargs.get('edge_index_rep')
            dep_rel_pos_edge_rep = kwargs.get('dep_rel_pos_edge_rep') if self.use_dep_rel_pos \
                else None
            word_rel_pos_edge_rep = kwargs.get('word_rel_pos_edge_rep') if self.use_word_rel_pos \
                else None
            deprel_edge_rep = kwargs.get('deprel_edge_rep') if self.use_deprel \
                else None
            deparc_edge_rep = kwargs.get('deparc_edge_rep')
            deprel_ext_edge_rep = kwargs.get('deprel_ext_edge_rep') if self.use_deprel_ext \
                else None
            path_len_edge_rep = kwargs.get('path_len_edge_rep') if (self.use_dep_path or self.use_dep_ext_path) \
                else None
            deprel_path_edge_rep = kwargs.get('deprel_path_edge_rep') if self.use_dep_path \
                else None
            deparc_path_edge_rep = kwargs.get('deparc_path_edge_rep') if self.use_dep_path \
                else None
            deprel_ext_path_edge_rep = kwargs.get('deprel_ext_path_edge_rep') if self.use_dep_ext_path \
                else None

            output, layer_outputs = self.encoder(
                inp=embs,
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
            )

        pred_ind_rep = kwargs.get('pred_ind_rep')

        # pooling
        pos_rep = kwargs.get('pos_rep')
        masks = pos_rep.eq(word.PAD)
        pred_ind_masks = ~pred_ind_rep.eq(self.pred_ind_yes_idx)
        pred_ind_masks = (pred_ind_masks | masks).unsqueeze(2)

        sent_out = pool(output, masks.unsqueeze(2))
        sent_out = reshape_according_to_sent_len(
            src=sent_out,
            ref=output,
            sent_len_rep=sent_len_rep
        )
        pred_ind_out = pool(output, pred_ind_masks)
        pred_ind_out = reshape_according_to_sent_len(
            src=pred_ind_out,
            ref=output,
            sent_len_rep=sent_len_rep
        )

        if self.use_sent_rep:
            outputs = torch.cat([sent_out, pred_ind_out, output], dim=-1)
        else:
            outputs = torch.cat([pred_ind_out, output], dim=-1)

        outputs = self.out_mlp(outputs)

        return outputs, sent_out
