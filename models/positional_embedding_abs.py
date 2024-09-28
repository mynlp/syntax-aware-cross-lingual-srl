import torch
import torch.nn as nn


class PositionalEmbeddingAbs(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbeddingAbs, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        tup = tuple(pos_seq.shape)
        new_dim = torch.prod(torch.tensor(pos_seq.shape))
        reshaped_pos_seq = pos_seq.view(new_dim)
        sinusoid_inp = torch.ger(reshaped_pos_seq, self.inv_freq)
        pos_emb = torch.zeros(new_dim.item(), self.demb).to(sinusoid_inp)
        pos_emb[:, ::2] = sinusoid_inp.sin()
        pos_emb[:, 1::2] = sinusoid_inp.cos()

        tup = tup + tuple([self.demb])

        return pos_emb.view(tup)
