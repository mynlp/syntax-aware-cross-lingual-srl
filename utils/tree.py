import numpy as np
from collections import deque
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from constants import word


def construct_dep_path_matrix(
    predec_matrix,
    length,
    deprel_matrix,
    deparc_matrix,
    deprel_ext_matrix
):
    deprel_path_matrix = np.full((length, length, length), word.PAD, dtype=int)
    deparc_path_matrix = np.full((length, length, length), word.PAD, dtype=int)
    deprel_ext_path_matrix = np.full((length, length, length), word.PAD, dtype=int)
    path_len_matrix = np.zeros((length, length), dtype=int)

    max_path_len = 0

    for i in range(length):
        for j in range(i, length):
            fr = predec_matrix[i, j]

            if fr == -9999 and i != j:
                path = []
                path_len_matrix[i, j] = 0
            else:
                path = [j]
                while fr != i and fr != -9999:
                    path.append(fr)
                    fr = predec_matrix[i, fr]
                path.append(i)
                path_len_matrix[i, j] = len(path) - 1

            max_path_len = max(max_path_len, path_len_matrix[i, j])

            fr = 0
            to = 1
            encoded_deprel_path = []
            encoded_deparc_path = []
            encoded_deprel_ext_path = []
            while to < len(path):
                encoded_deprel_path.append(deprel_matrix[path[to], path[fr]])
                encoded_deparc_path.append(deparc_matrix[path[to], path[fr]])
                encoded_deprel_ext_path.append(deprel_ext_matrix[path[to], path[fr]])
                fr = to
                to += 1
            deprel_path_matrix[i, j, :len(encoded_deprel_path)] = encoded_deprel_path
            deparc_path_matrix[i, j, :len(encoded_deparc_path)] = encoded_deparc_path
            deprel_ext_path_matrix[i, j, :len(encoded_deprel_ext_path)] = encoded_deprel_ext_path

            if i != j:
                path_len_matrix[j, i] = path_len_matrix[i, j]
                path.reverse()
                fr = 0
                to = 1
                encoded_deprel_path = []
                encoded_deparc_path = []
                encoded_deprel_ext_path = []
                while to < len(path):
                    encoded_deprel_path.append(deprel_matrix[path[to], path[fr]])
                    encoded_deparc_path.append(deparc_matrix[path[to], path[fr]])
                    encoded_deprel_ext_path.append(deprel_ext_matrix[path[to], path[fr]])
                    fr = to
                    to += 1
                deprel_path_matrix[j, i, :len(encoded_deprel_path)] = encoded_deprel_path
                deparc_path_matrix[j, i, :len(encoded_deparc_path)] = encoded_deparc_path
                deprel_ext_path_matrix[j, i, :len(encoded_deprel_ext_path)] = encoded_deprel_ext_path

    deprel_path_matrix_final = np.full((length, length, max_path_len), word.PAD, dtype=int)
    deparc_path_matrix_final = np.full((length, length, max_path_len), word.PAD, dtype=int)
    deprel_ext_path_matrix_final = np.full((length, length, max_path_len), word.PAD, dtype=int)

    deprel_path_matrix_final[:, :, :max_path_len] = deprel_path_matrix[:, :, :max_path_len]
    deparc_path_matrix_final[:, :, :max_path_len] = deparc_path_matrix[:, :, :max_path_len]
    deprel_ext_path_matrix_final[:, :, :max_path_len] = deprel_ext_path_matrix[:, :, :max_path_len]

    return deprel_path_matrix_final, deparc_path_matrix_final, deprel_ext_path_matrix_final, path_len_matrix


def construct_dep_matrix_from_tree(
        length,
        root,
        model,
        upb_version
):
    deprel_voc = model.deprel_voc_by_version[upb_version]
    deprel_ext_voc = model.deprel_ext_voc_by_version[upb_version]
    deprel_matrix = np.full((length, length), deprel_voc[word.NO_RELATION_DEPREL], dtype=int)
    deprel_ext_matrix = np.full((length, length), deprel_ext_voc[word.NO_RELATION_DEPREL], dtype=int)
    deparc_matrix = np.full((length, length), model.deparc_voc[word.deparc_map['norelation']], dtype=int)

    q = deque()
    q.append(root)
    indices = []

    while q:
        node = q.popleft()
        indices.append(node.idx)
        for child in node.children:
            deprel_matrix[child.idx, node.idx] = deprel_voc[child.deprel]
            deprel_matrix[node.idx, child.idx] = deprel_voc[child.deprel]
            deprel_ext_matrix[child.idx, node.idx] = deprel_ext_voc[child.deprel]
            deprel_ext_matrix[node.idx, child.idx] = deprel_ext_voc[f'{child.deprel}~']
            deparc_matrix[child.idx, node.idx] = model.deparc_voc[word.deparc_map['align']]
            deparc_matrix[node.idx, child.idx] = model.deparc_voc[word.deparc_map['opposite']]
            q.append(child)

    for idx in indices:
        deprel_ext_matrix[idx, idx] = deprel_ext_voc[word.SELF_DEPREL]
        deprel_matrix[idx, idx] = deprel_voc[word.SELF_DEPREL]
        deparc_matrix[idx, idx] = model.deparc_voc[word.deparc_map['self']]

    return deprel_matrix, deparc_matrix, deprel_ext_matrix


def construct_adjacency_matrix_from_tree(
        length,
        root,
        is_directed,
        is_self_loop
):
    adj_matrix = np.zeros((length, length), dtype=np.float32)

    q = deque()
    q.append(root)
    indices = []

    while q:
        node = q.popleft()
        indices.append(node.idx)
        for child in node.children:
            adj_matrix[child.idx, node.idx] = 1
            q.append(child)

    if not is_directed:
        adj_matrix = adj_matrix + adj_matrix.T

    if is_self_loop:
        for idx in indices:
            adj_matrix[idx, idx] = 1

    return adj_matrix


def construct_distance_matrix_from_adjacency_matrix(
        adj_matrix,
        is_directed,
        is_self_loop
):
    dist_matrix, predec_matrix = shortest_path(
        csgraph=csr_matrix(adj_matrix),
        directed=is_directed,
        return_predecessors=True
    )

    if is_self_loop:
        np.fill_diagonal(dist_matrix, 1)

    return dist_matrix, predec_matrix
