import torch
import torch.nn as nn
from constants import word
from models.pretrained_word_embedding import PretrainedWordEmbedding
from models.positional_embedding_abs import PositionalEmbeddingAbs


class Embedder(nn.Module):
    def __init__(self, args):
        super(Embedder, self).__init__()

        self.pretrained_word_emb = PretrainedWordEmbedding(args)
        self.pos_emb = None
        self.deprel_emb = None
        self.word_abs_position_emb = None
        self.dep_abs_position_emb = None
        self.pred_ind_emb = None
        self.use_positional_embedding = args.use_positional_embedding

        if args.pos_dim > 0:
            self.pos_emb = nn.Embedding(
                args.pos_size,
                args.pos_dim,
                padding_idx=word.PAD
            )
        if args.deprel_dim > 0:
            self.deprel_emb = nn.Embedding(
                args.deprel_size,
                args.deprel_dim,
                padding_idx=word.PAD
            )
        if args.abs_position_dim > 0:
            if args.use_dep_abs_position:
                if self.use_positional_embedding:
                    self.dep_abs_position_emb = PositionalEmbeddingAbs(
                        args.abs_position_dim
                    )
                else:
                    self.dep_abs_position_emb = nn.Embedding(
                        word.DUMMY_MAX_LEN + 1,
                        args.abs_position_dim,
                        padding_idx=word.DUMMY_MAX_LEN
                    )
            if args.use_word_abs_position:
                if self.use_positional_embedding:
                    self.word_abs_position_emb = PositionalEmbeddingAbs(
                        args.abs_position_dim
                    )
                else:
                    self.word_abs_position_emb = nn.Embedding(
                        word.DUMMY_MAX_LEN + 1,
                        args.abs_position_dim,
                        padding_idx=word.DUMMY_MAX_LEN
                    )

        if args.pred_ind_dim > 0:
            self.pred_ind_emb = nn.Embedding(
                args.pred_ind_size,
                args.pred_ind_dim,
                padding_idx=word.PAD
            )

        self.drop_in = nn.Dropout(args.emb_dropout)
        emb_dims = [
            self.pretrained_word_emb.out_dim,
            args.pos_dim,
            args.deprel_dim,
            args.pred_ind_dim
        ]

        if args.abs_position_dim > 0:
            if args.use_dep_abs_position:
                emb_dims.append(args.abs_position_dim)
            if args.use_word_abs_position:
                emb_dims.append(args.abs_position_dim)

        args.emb_dim = sum(emb_dims)

    def forward(self, **kwargs):
        sent_len_rep = kwargs.get('sent_len_rep')
        we_input_id_rep = kwargs.get('we_input_id_rep')
        we_offset_rep = kwargs.get('we_offset_rep')
        we_len_rep = kwargs.get('we_len_rep')
        pred_ind_rep = kwargs.get('pred_ind_rep')
        pos_rep = kwargs.get('pos_rep')
        word_abs_position_rep = kwargs.get('word_abs_position_rep')
        dep_abs_position_rep = kwargs.get('dep_abs_position_rep')
        deprel_rep = kwargs.get('deprel_rep')
        mask = pos_rep.eq(word.PAD).unsqueeze(-1).bool()

        embs = [
            self.pretrained_word_emb(
                sentence_lengths=sent_len_rep,
                input_ids=we_input_id_rep,
                word_offsets=we_offset_rep,
                tokenized_sentence_lengths=we_len_rep
            )
        ]
        if self.pred_ind_emb is not None:
            embs.append(self.pred_ind_emb(pred_ind_rep))
        if self.pos_emb is not None:
            embs.append(self.pos_emb(pos_rep))
        if self.word_abs_position_emb is not None:
            res = self.word_abs_position_emb(word_abs_position_rep)
            if self.use_positional_embedding:
                res = res.masked_fill(mask, 0)
            embs.append(res)
        if self.dep_abs_position_emb is not None:
            res = self.dep_abs_position_emb(dep_abs_position_rep)
            if self.use_positional_embedding:
                res = res.masked_fill(mask, 0)
            embs.append(res)
        if self.deprel_emb is not None:
            embs.append(self.deprel_emb(deprel_rep))

        embs = torch.cat(embs, dim=-1)
        embs = self.drop_in(embs)
        return embs
