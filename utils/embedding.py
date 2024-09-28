import torch


def get_last_embeddings(
        hidden_states
):
    return hidden_states[-1]


def get_last_four_cat_embeddings(
        hidden_states
):
    return torch.cat((hidden_states[-4], hidden_states[-3], hidden_states[-2], hidden_states[-1]), dim=-1)


def extract_embedding_layer(
        layer_extraction_mode,
        hidden_states
):
    if layer_extraction_mode == 'last':
        get_embeddings_function = get_last_embeddings
    elif layer_extraction_mode == 'last_four_cat':
        get_embeddings_function = get_last_four_cat_embeddings
    else:
        raise Exception(f'Layer extraction mode is unknown. Mode: {layer_extraction_mode}.')

    return get_embeddings_function(hidden_states)
