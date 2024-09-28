import numpy as np
from constants import word


class Instance:
    def __init__(self, sentence, pas):
        self.sentence = sentence
        self.pas = pas
        self.pred_ind_ids = None
        self.sem_role_ids = None
        self.pred_adj_matrix = None

    def initialize(self, model):
        sentence_len = len(self.sentence.words)
        dist_matrix = self.sentence.dist_matrix

        self.pred_ind_ids = [model.pred_ind_voc[word.pred_ind_map['no']]] * sentence_len
        yes_idx = model.pred_ind_voc[word.pred_ind_map['yes']]

        for idx in self.pas['V']:
            self.pred_ind_ids[idx] = yes_idx

        pred_ind_ids_ = np.array(self.pred_ind_ids)
        self.pred_adj_matrix = np.zeros((sentence_len, sentence_len), dtype=int)
        self.pred_adj_matrix[dist_matrix == 1] = 1
        self.pred_adj_matrix[pred_ind_ids_ == yes_idx, :] = 1
        self.pred_adj_matrix[:, pred_ind_ids_ == yes_idx] = 1

        self.sem_role_ids = [model.semantic_role_voc[word.NO_RELATION_SEMROLE]] * sentence_len
        for key in self.pas:
            if key == 'V':
                continue

            for idx in self.pas[key]:
                assert self.sem_role_ids[idx] == model.semantic_role_voc[word.NO_RELATION_SEMROLE]
                self.sem_role_ids[idx] = model.semantic_role_voc[key]
