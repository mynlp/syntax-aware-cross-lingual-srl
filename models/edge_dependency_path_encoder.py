import torch
import torch.nn as nn
from constants import word
from torch.nn.utils.rnn import pack_padded_sequence


class EdgeDependencyPathEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        dropout,
        deprel_size,
        deparc_size,
        deprel_ext_size,
        deprel_edge_dim,
        deparc_edge_dim,
        deprel_ext_edge_dim,
        use_dep_path,
        use_dep_ext_path,
        sum_dep_path
    ):
        super(EdgeDependencyPathEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.use_dep_ext_path = use_dep_ext_path
        self.sum_dep_path = sum_dep_path

        if not self.sum_dep_path:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                bias=True,
                batch_first=True,
                dropout=dropout,
                bidirectional=False
            )

        if self.use_dep_ext_path:
            self.deprel_ext_embeddings = nn.Embedding(
                deprel_ext_size,
                deprel_ext_edge_dim,
                padding_idx=word.PAD
            )
        else:
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

    def count_parameters(self):
        params = list(self.layer.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(
            self,
            deprel_path_edge,
            deparc_path_edge,
            path_len_edge,
            deprel_ext_path_edge
    ):
        seq_path_len = path_len_edge
        seq_path_len, sorted_indices = seq_path_len.sort(0, descending=True)

        if self.use_dep_ext_path:
            seq_deprel_ext_path = deprel_ext_path_edge[sorted_indices]
            seq_path = self.deprel_ext_embeddings(seq_deprel_ext_path)
        else:
            seq_deprel_path = deprel_path_edge[sorted_indices]
            seq_deparc_path = deparc_path_edge[sorted_indices]

            seq_deprel_path = self.deprel_embeddings(seq_deprel_path)
            seq_deparc_path = self.deparc_embeddings(seq_deparc_path)
            seq_path = torch.cat((seq_deprel_path, seq_deparc_path), -1)

        if self.sum_dep_path:
            out = torch.sum(seq_path, dim=-2)
            out[sorted_indices] = out.clone()
        else:
            packed_input = pack_padded_sequence(seq_path, seq_path_len.cpu(), batch_first=True)
            packed_output, (ht, ct) = self.lstm(packed_input)

            out = ht[-1]
            out[sorted_indices] = out.clone()
            # seq_path_len[sorted_indices] = seq_path_len.clone()

        return out
