import os
import torch
import torch.nn as nn
from constants import embedding
from utils.embedding import extract_embedding_layer
from transformers import AutoTokenizer, AutoConfig, AutoModel


class PretrainedWordEmbedding(nn.Module):
    def __init__(
            self,
            args
    ):
        super(PretrainedWordEmbedding, self).__init__()
        self.pos_extraction_mode = args.pretrained_we_pos_extraction_mode
        self.layer_extraction_mode = args.pretrained_we_layer_extraction_mode
        self.is_transform = args.we_out_dim != 0
        self.is_fine_tuned = args.fine_tuned_we
        self.model_name = args.pretrained_we_model_name

        assert self.pos_extraction_mode in embedding.pos_extraction_mode
        assert self.layer_extraction_mode in embedding.layer_extraction_mode
        assert self.model_name in embedding.model_name

        if self.pos_extraction_mode == 'left':
            self.get_full_word_embeddings_from_sub_word_embeddings = \
                self.get_left_word_embeddings_from_sub_word_embeddings
        elif self.pos_extraction_mode == 'right':
            self.get_full_word_embeddings_from_sub_word_embeddings = \
                self.get_right_word_embeddings_from_sub_word_embeddings
        else:
            self.get_full_word_embeddings_from_sub_word_embeddings = \
                self.get_avg_word_embeddings_from_sub_word_embeddings

        if self.layer_extraction_mode == 'last_four_cat':
            self.in_dim = 4 * embedding.PRETRAINED_EMBEDDING_SIZE
        else:
            self.in_dim = embedding.PRETRAINED_EMBEDDING_SIZE

        if self.is_transform:
            self.out_dim = embedding.PRETRAINED_EMBEDDING_SIZE if args.we_out_dim == -1 else args.we_out_dim
        else:
            self.out_dim = self.in_dim

        sub_folder_name = 'xlm' if 'xlm' in self.model_name else 'bert'
        self.model_dir = f"{os.getenv('WORD_EMB_DIR')}/{sub_folder_name}/{self.model_name}" \
            if os.getenv('WORD_EMB_DIR') else self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        config = AutoConfig.from_pretrained(self.model_dir, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(self.model_dir, config=config)

        if self.is_transform:
            self.projection = nn.Linear(self.in_dim, self.out_dim)

    def eval(self):
        self.is_fine_tuned = False
        self.model.eval()

    @property
    def padding_id(self):
        return self.tokenizer.pad_token_id

    def get_input_ids(
            self,
            tokenized_sentence
    ):
        return self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

    def tokenize_sentence(
            self,
            words
    ):
        tokenized_sentence = [self.tokenizer.cls_token]
        offsets = []
        unk_token_count = 0

        for word in words:
            tokenized_word = self.tokenizer.tokenize(word)

            if len(tokenized_word) == 0:
                tokenized_word = [self.tokenizer.unk_token]
                unk_token_count += 1

            tokenized_sentence.extend(tokenized_word)
            offsets.append(len(tokenized_word))

        if unk_token_count > 0:
            print(f'There are {unk_token_count} unknown tokens.')

        tokenized_sentence.append(self.tokenizer.sep_token)

        return tokenized_sentence, offsets

    def get_embeddings(
            self,
            input_ids
    ):
        if self.is_fine_tuned:
            outputs = self.model(input_ids=input_ids)

            # Dimension: layer, batch, token, embedding
            hidden_states = outputs.hidden_states
            hidden_states = torch.stack(hidden_states, dim=0)
        else:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)

                # Dimension: layer, batch, token, embedding
                hidden_states = outputs.hidden_states
                hidden_states = torch.stack(hidden_states, dim=0)

        return extract_embedding_layer(
            layer_extraction_mode=self.layer_extraction_mode,
            hidden_states=hidden_states
        )

    def get_avg_word_embeddings_from_sub_word_embeddings(
            self,
            word_offsets,
            sentence_length,
            tokenized_sentence_length,
            embedded_sentence
    ):
        i_bert = 1  # Skip [CLS] token
        new_embedded_sentence = []

        for i_offset in range(sentence_length):
            word_offset = word_offsets[i_offset]

            new_embedded_sentence.append(torch.mean(embedded_sentence[i_bert:i_bert + word_offset], dim=0))

            i_bert += word_offset

        if i_bert != tokenized_sentence_length - 1:  # Omit [SEP] token
            raise Exception(f'Fail to get full word sentence embeddings. '
                            f'i_bert: {i_bert}. '
                            f'Tokenized sentence len: {tokenized_sentence_length}.')

        new_embedded_sentence = torch.stack(new_embedded_sentence)

        return new_embedded_sentence

    def get_right_word_embeddings_from_sub_word_embeddings(
            self,
            word_offsets,
            sentence_length,
            tokenized_sentence_length,
            embedded_sentence
    ):
        i_bert = 1  # Skip [CLS] token
        chosen_indices = []

        for i_offset in range(sentence_length):
            word_offset = word_offsets[i_offset]

            i_bert += word_offset

            chosen_indices.append(i_bert - 1)

        if i_bert != tokenized_sentence_length - 1:  # Omit [SEP] token
            raise Exception(f'Fail to get full word sentence embeddings. '
                            f'i_bert: {i_bert}. '
                            f'Tokenized sentence len: {tokenized_sentence_length}.')

        return embedded_sentence.index_select(0, torch.tensor(chosen_indices, device=embedded_sentence.device))

    def get_left_word_embeddings_from_sub_word_embeddings(
            self,
            word_offsets,
            sentence_length,
            tokenized_sentence_length,
            embedded_sentence
    ):
        i_bert = 1  # Skip [CLS] token
        chosen_indices = []

        for i_offset in range(sentence_length):
            word_offset = word_offsets[i_offset]

            chosen_indices.append(i_bert)

            i_bert += word_offset

        if i_bert != tokenized_sentence_length - 1:  # Omit [SEP] token
            raise Exception(f'Fail to get full word sentence embeddings. '
                            f'i_bert: {i_bert}. '
                            f'Tokenized sentence len: {tokenized_sentence_length}.')

        return embedded_sentence.index_select(0, torch.tensor(chosen_indices, device=embedded_sentence.device))

    def forward(
            self,
            sentence_lengths,
            input_ids,
            word_offsets,
            tokenized_sentence_lengths
    ):
        embedded_sentences = self.get_embeddings(input_ids=input_ids)

        _, max_sent_len = word_offsets.size()
        batch_size, _, num_feature = embedded_sentences.size()
        new_embedded_sentences = torch.zeros((batch_size, max_sent_len, num_feature)).to(embedded_sentences)

        for i in range(batch_size):
            sentence_length = sentence_lengths[i]

            new_embedded_sentences[i, :sentence_length] = self.get_full_word_embeddings_from_sub_word_embeddings(
                word_offsets=word_offsets[i],
                sentence_length=sentence_length,
                tokenized_sentence_length=tokenized_sentence_lengths[i],
                embedded_sentence=embedded_sentences[i]
            )

        if self.is_transform:
            new_embedded_sentences = self.projection(new_embedded_sentences)

        return new_embedded_sentences
