from constants import word


class Vocabulary(object):
    def __init__(self, token_list=[], vocab_type=''):
        self.tok2ind = dict()
        self.ind2tok = dict()
        self._pad_word = word.PAD_WORD
        self._unk_word = word.UNK_WORD

        if vocab_type.lower() == 'bert':
            self._pad_word = word.BERT_PAD_WORD
            self._unk_word = word.BERT_UNK_WORD
        elif vocab_type.lower() == 'regular':
            self._pad_word = word.PAD_WORD
            self._unk_word = word.UNK_WORD
            self.tok2ind[word.PAD_WORD] = word.PAD
            self.tok2ind[word.UNK_WORD] = word.UNK
            self.ind2tok[word.PAD] = word.PAD_WORD
            self.ind2tok[word.UNK] = word.UNK_WORD

        self.add_tokens(token_list)

    @property
    def pad_word(self):
        return self._pad_word

    @property
    def unk_word(self):
        return self._unk_word

    @property
    def pad_idx(self):
        return self.tok2ind[self._pad_word]

    @property
    def unk_idx(self):
        return self.tok2ind[self._unk_word]

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return key in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.unk_word)
        elif type(key) == str:
            return self.tok2ind.get(key,
                                    self.tok2ind.get(self.unk_word))
        else:
            raise RuntimeError('Invalid key type.')

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def add_tokens(self, token_list):
        assert isinstance(token_list, list)
        for token in token_list:
            self.add(token)

    def tokens(self):
        """Get dictionary tokens.
        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {self.pad_word, self.unk_word}]
        return tokens

    def load(self, vocab):
        for token, index in vocab.items():
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def remove(self, key):
        if key in self.tok2ind:
            ind = self.tok2ind[key]
            del self.tok2ind[key]
            del self.ind2tok[ind]
            return True
        return False
