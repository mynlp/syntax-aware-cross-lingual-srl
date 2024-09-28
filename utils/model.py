import torch
from torch_scatter import scatter_max
from constants import word, model


def is_trans(network_types):
    return (model.network_type['trans'] in network_types) or (model.network_type['trans_gaan'] in network_types)


def pool(h, mask):
    h = h.masked_fill(mask.bool(), -word.INFINITY_NUMBER)
    return torch.max(h, 1)[0]


def reshape_according_to_sent_len(
    src,
    ref,
    sent_len_rep
):
    [batch_size, max_len, num_features] = ref.size()
    tgt = torch.zeros(
        [batch_size, max_len, num_features],
        dtype=ref.dtype,
        device=ref.device
    )

    for i in range(batch_size):
        tgt[i, :sent_len_rep[i]] = src[i].unsqueeze(0).repeat(sent_len_rep[i], 1)

    return tgt


def shape(
    x,
    dim,
    batch_size,
    num_heads
):
    """  projection """
    return x.view(batch_size, -1, num_heads, dim).transpose(1, 2)


def unshape(
    x,
    dim,
    batch_size,
    num_heads
):
    """  compute context """
    return x.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * dim)


def generate_relative_positions_matrix(length):
    range_vec = torch.arange(length)
    range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
    distance_mat = range_mat.transpose(0, 1) - range_mat

    return distance_mat


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)  # qlen x bsz x nhead x dim
    x_t_r = x_t.reshape(length, heads * batch_size, -1)  # qlen x (bsz * nhead) x dim
    if transpose:
        z_t = z.transpose(1, 2)  # klen x dim x klen
        # qlen must be either 1 or qlen = klen
        x_tz_matmul = torch.matmul(x_t_r, z_t)  # klen x (bsz * nhead) x klen
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    # x_tz_matmul_r: qlen x bsz x nhead x klen
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    # x_tz_matmul_r_t: bsz x nhead x qlen x klen
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


def apply_gate_to_weight_heads_in_trans(
    neighbor,
    key_len,
    mask,
    adj_mask,
    original_value
):
    # bsz x key_len x key_len x neighbor_dim
    complete_neighbor = neighbor.unsqueeze(1).repeat(1, key_len, 1, 1)
    # bsz x query_len x key_len x 1
    final_mask = mask.squeeze(1).unsqueeze(-1)
    complete_neighbor = complete_neighbor.masked_fill(final_mask.bool(), -word.INFINITY_NUMBER)

    if adj_mask is not None:
        # bsz x query_len x key_len x 1
        final_adj_mask = adj_mask[:, 0, :, :].unsqueeze(-1)
        complete_neighbor = complete_neighbor.masked_fill(~final_adj_mask.bool(), -word.INFINITY_NUMBER)
    else:
        final_adj_mask = None

    # bsz x key_len x neighbor_dim
    max_gate = torch.max(complete_neighbor, 2)[0]

    complete_neighbor = original_value.unsqueeze(1).repeat(1, key_len, 1, 1)
    complete_neighbor = complete_neighbor.masked_fill(final_mask.bool(), 0)

    if final_adj_mask is not None:
        complete_neighbor = complete_neighbor.masked_fill(~final_adj_mask.bool(), 0)

    # bsz x key_len x d_model
    sum_gate = torch.sum(complete_neighbor, 2)
    # bsz x key_len x key_len
    invert_neighbor_sum = final_mask.squeeze(-1).bool()

    if final_adj_mask is not None:
        invert_neighbor_sum = invert_neighbor_sum | ~final_adj_mask.squeeze(-1).bool()

    # bsz x key_len x key_len
    neighbor = (~invert_neighbor_sum).long()
    # bsz x key_len x 1
    neighbor_count = torch.sum(neighbor, 2).unsqueeze(-1)

    avg_gate = sum_gate / (neighbor_count + (1 / word.INFINITY_NUMBER))

    return max_gate, avg_gate


def explicit_broadcast(this, other):
    # Append singleton dimensions until this.dim() == other.dim()
    for _ in range(this.dim(), other.dim()):
        this = this.unsqueeze(-1)

    # Explicitly expand so that shapes are the same
    return this.expand_as(other)


def sum_edge_scores_neighborhood_aware(
    scores_per_edge,
    tgt_index,
    neighbor_sum
):
    tgt_index_broadcasted = explicit_broadcast(tgt_index, scores_per_edge)

    neighbor_sum.scatter_add_(0, tgt_index_broadcasted, scores_per_edge)


def max_edge_scores_neighborhood_aware(
    scores_per_edge,
    tgt_index,
    neighbor_max
):
    tgt_index_broadcasted = explicit_broadcast(tgt_index, scores_per_edge)

    scatter_max(scores_per_edge, tgt_index_broadcasted, dim=0, out=neighbor_max)


def apply_gate_to_weight_heads_in_gnn(
    neighbor,
    num_nodes,
    model_dim,
    tgt_index,
    num_edges,
    projected_neighbor,
    neighbor_dim
):
    neighbor_sum_gate = torch.zeros((num_nodes, model_dim)).to(neighbor)
    sum_edge_scores_neighborhood_aware(
        scores_per_edge=neighbor,
        tgt_index=tgt_index,
        neighbor_sum=neighbor_sum_gate
    )

    # edge_size x 1
    edge = torch.ones((num_edges, 1)).to(neighbor)
    neighbor_count_gate = torch.zeros((num_nodes, 1)).to(neighbor)
    sum_edge_scores_neighborhood_aware(
        scores_per_edge=edge,
        tgt_index=tgt_index,
        neighbor_sum=neighbor_count_gate
    )

    # node_size x model_dim
    avg_gate = neighbor_sum_gate / (neighbor_count_gate + (1 / word.INFINITY_NUMBER))

    max_gate = torch.zeros((num_nodes, neighbor_dim)).to(projected_neighbor)

    max_edge_scores_neighborhood_aware(
        scores_per_edge=projected_neighbor,
        tgt_index=tgt_index,
        neighbor_max=max_gate
    )

    return max_gate, avg_gate
