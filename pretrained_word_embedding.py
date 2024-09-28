import torch
from models.pretrained_word_embedding import PretrainedWordEmbedding

class Args:
    def __init__(
            self
    ):
        self.pretrained_we_pos_extraction_mode = 'left'
        self.pretrained_we_layer_extraction_mode = 'last'
        self.we_out_dim = 0
        self.fine_tuned_we = False
        self.pretrained_we_model_name = 'xlm-roberta-base'


if __name__ == '__main__':
    emb = PretrainedWordEmbedding(Args())

    stc1, offset1 = emb.tokenize_sentence(['I', 'am', 'exhausted', '.'])
    stc2, offset2 = emb.tokenize_sentence(['I', 'want', 'to', 'go', 'to', 'bed', '.'])

    print(stc1)
    print(stc2)
    maximum = max(len(stc1), len(stc2))

    input_ids = torch.tensor([
        emb.get_input_ids(stc1) + [emb.padding_id] * (maximum - len(stc1)),
        emb.get_input_ids(stc2) + [emb.padding_id] * (maximum - len(stc2))
    ])

    print('input_ids', input_ids)

    maximum = max(len(offset1), len(offset2))

    sentence_lengths = torch.tensor([
        len(offset1),
        len(offset2)
    ])

    print('sentence_lengths', sentence_lengths)

    word_offsets = torch.tensor([
        offset1 + [0] * (maximum - len(offset1)),
        offset2 + [0] * (maximum - len(offset2))
    ])

    tokenized_sentence_lengths = torch.tensor([
        len(stc1),
        len(stc2)
    ])

    print('tokenized_sentence_lengths', tokenized_sentence_lengths)

    res = emb(
          sentence_lengths=sentence_lengths,
          input_ids=input_ids,
          word_offsets=word_offsets,
          tokenized_sentence_lengths=tokenized_sentence_lengths
    )

    print(res.shape)
