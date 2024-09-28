import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as f

from constants import word
from utils.file import load_array_from_file
from classes.vocabulary import Vocabulary
from models.gtn_classifier import GTNClassifier
from utils.model import is_trans


class SemanticRoleLabeler:
    def __init__(
            self,
            args,
            state_dict=None,
            vocab=None
    ):
        self.args = args

        self.deprel_voc_by_version = dict()
        self.deprel_ext_voc_by_version = dict()
        self.pos_voc = None
        self.semantic_role_voc = None
        self.pred_ind_voc = None
        self.deparc_voc = None

        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        self.is_trans = is_trans(self.args.network_type)

        self.load_vocabulary(vocab=vocab)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.semantic_role_voc.pad_idx
        )

        args.pred_ind_yes_idx = self.pred_ind_voc[word.pred_ind_map['yes']]
        args.deparc_voc = self.deparc_voc
        args.deprel_ext_voc = self.deprel_ext_voc_by_version[args.upb_version]

        self.network = GTNClassifier(args)

        if state_dict is not None:
            self.get_network().load_state_dict(state_dict)

    def get_network(self):
        if self.parallel:
            return self.network.module
        else:
            return self.network

    def load_vocabulary(self, vocab=None):
        if vocab:
            self.deprel_voc_by_version = vocab['deprel_voc_by_version']
            self.deprel_ext_voc_by_version = vocab['deprel_ext_voc_by_version']
            self.pos_voc = vocab['pos_voc']
            self.semantic_role_voc = vocab['semantic_role_voc']
            self.pred_ind_voc = vocab['pred_ind_voc']
            self.deparc_voc = vocab['deparc_voc']
        else:
            self.deprel_ext_voc_by_version[1] = Vocabulary(
                token_list=load_array_from_file('vocabs/deprel1_ext.txt'),
                vocab_type='regular'
            )
            self.deprel_ext_voc_by_version[2] = Vocabulary(
                token_list=load_array_from_file('vocabs/deprel2_ext.txt'),
                vocab_type='regular'
            )
            self.deprel_voc_by_version[1] = Vocabulary(
                token_list=load_array_from_file('vocabs/deprel1.txt'),
                vocab_type='regular'
            )
            self.deprel_voc_by_version[2] = Vocabulary(
                token_list=load_array_from_file('vocabs/deprel2.txt'),
                vocab_type='regular'
            )
            self.pos_voc = Vocabulary(
                token_list=load_array_from_file('vocabs/pos_tag.txt'),
                vocab_type='regular'
            )
            self.semantic_role_voc = Vocabulary(
                token_list=load_array_from_file('vocabs/semantic_role.txt'),
                vocab_type='regular'
            )
            self.deparc_voc = Vocabulary(
                token_list=load_array_from_file('vocabs/deparc.txt'),
                vocab_type='regular'
            )
            self.pred_ind_voc = Vocabulary(
                token_list=load_array_from_file('vocabs/pred_ind.txt'),
                vocab_type='regular'
            )

        self.args.deprel_size = max(len(self.deprel_voc_by_version[1]), len(self.deprel_voc_by_version[2]))
        self.args.deprel_ext_size = max(len(self.deprel_ext_voc_by_version[1]), len(self.deprel_ext_voc_by_version[2]))
        self.args.pos_size = len(self.pos_voc)
        self.args.label_size = len(self.semantic_role_voc)
        self.args.deparc_size = len(self.deparc_voc)
        self.args.pred_ind_size = len(self.pred_ind_voc)

    def init_optimizer(self, state_dict=None, use_gpu=True):
        parameters = [p for p in self.get_network().parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                parameters,
                self.args.learning_rate,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                parameters,
                self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                parameters,
                self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        else:
            raise RuntimeError(f'Unsupported optimizer: {self.args.optimizer}.')

        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)
            # FIXME: temp soln - https://github.com/pytorch/pytorch/issues/2830
            if use_gpu:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

    def update(self, ex):
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        sent_len_rep = ex['sent_len_rep']
        we_len_rep = ex['we_len_rep']
        pos_rep = ex['pos_rep']
        deprel_rep = ex['deprel_rep']
        pred_ind_rep = ex['pred_ind_rep']
        sem_role_rep = ex['sem_role_rep']
        we_offset_rep = ex['we_offset_rep']
        we_input_id_rep = ex['we_input_id_rep']
        word_abs_position_rep = ex['word_abs_position_rep']
        dep_abs_position_rep = ex['dep_abs_position_rep']

        adj_mask_rep = ex['adj_mask_rep'] if self.is_trans else None
        fc_mask_rep = ex['fc_mask_rep'] if self.is_trans else None
        pred_adj_mat_rep = ex['pred_adj_mat_rep'] if self.is_trans else None
        deprel_path_mat_rep = ex['deprel_path_mat_rep'] if self.is_trans else None
        deparc_path_mat_rep = ex['deparc_path_mat_rep'] if self.is_trans else None
        deprel_ext_path_mat_rep = ex['deprel_ext_path_mat_rep'] if self.is_trans else None
        path_len_mat_rep = ex['path_len_mat_rep'] if self.is_trans else None
        dep_rel_pos_mat_rep = ex['dep_rel_pos_mat_rep'] if self.is_trans else None
        word_rel_pos_mat_rep = ex['word_rel_pos_mat_rep'] if self.is_trans else None
        deprel_mat_rep = ex['deprel_mat_rep'] if self.is_trans else None
        deparc_mat_rep = ex['deparc_mat_rep'] if self.is_trans else None
        deprel_ext_mat_rep = ex['deprel_ext_mat_rep'] if self.is_trans else None

        deprel_path_edge_rep = None if self.is_trans else ex['deprel_path_edge_rep']
        deparc_path_edge_rep = None if self.is_trans else ex['deparc_path_edge_rep']
        deprel_ext_path_edge_rep = None if self.is_trans else ex['deprel_ext_path_edge_rep']
        path_len_edge_rep = None if self.is_trans else ex['path_len_edge_rep']
        dep_rel_pos_edge_rep = None if self.is_trans else ex['dep_rel_pos_edge_rep']
        word_rel_pos_edge_rep = None if self.is_trans else ex['word_rel_pos_edge_rep']
        deprel_edge_rep = None if self.is_trans else ex['deprel_edge_rep']
        deparc_edge_rep = None if self.is_trans else ex['deparc_edge_rep']
        edge_index_rep = None if self.is_trans else ex['edge_index_rep']
        edge_len_rep = None if self.is_trans else ex['edge_len_rep']
        dist_edge_rep = None if self.is_trans else ex['dist_edge_rep']
        deprel_ext_edge_rep = None if self.is_trans else ex['deprel_ext_edge_rep']

        if self.use_cuda:
            sent_len_rep = sent_len_rep.cuda(non_blocking=True)
            we_len_rep = we_len_rep.cuda(non_blocking=True)
            pos_rep = pos_rep.cuda(non_blocking=True)
            deprel_rep = deprel_rep.cuda(non_blocking=True)
            pred_ind_rep = pred_ind_rep.cuda(non_blocking=True)
            sem_role_rep = sem_role_rep.cuda(non_blocking=True)
            we_offset_rep = we_offset_rep.cuda(non_blocking=True)
            we_input_id_rep = we_input_id_rep.cuda(non_blocking=True)
            word_abs_position_rep = word_abs_position_rep.cuda(non_blocking=True)
            dep_abs_position_rep = dep_abs_position_rep.cuda(non_blocking=True)

            if self.is_trans:
                adj_mask_rep = adj_mask_rep.cuda(non_blocking=True)
                fc_mask_rep = fc_mask_rep.cuda(non_blocking=True)
                pred_adj_mat_rep = pred_adj_mat_rep.cuda(non_blocking=True)
                dep_rel_pos_mat_rep = dep_rel_pos_mat_rep.cuda(non_blocking=True)
                word_rel_pos_mat_rep = word_rel_pos_mat_rep.cuda(non_blocking=True)
                deprel_mat_rep = deprel_mat_rep.cuda(non_blocking=True)
                deparc_mat_rep = deparc_mat_rep.cuda(non_blocking=True)
                path_len_mat_rep = path_len_mat_rep.cuda(non_blocking=True)
                deprel_path_mat_rep = deprel_path_mat_rep.cuda(non_blocking=True)
                deparc_path_mat_rep = deparc_path_mat_rep.cuda(non_blocking=True)
                deprel_ext_mat_rep = deprel_ext_mat_rep.cuda(non_blocking=True)
                deprel_ext_path_mat_rep = deprel_ext_path_mat_rep.cuda(non_blocking=True)
            else:
                deprel_path_edge_rep = deprel_path_edge_rep.cuda(non_blocking=True)
                deparc_path_edge_rep = deparc_path_edge_rep.cuda(non_blocking=True)
                path_len_edge_rep = path_len_edge_rep.cuda(non_blocking=True)
                dep_rel_pos_edge_rep = dep_rel_pos_edge_rep.cuda(non_blocking=True)
                word_rel_pos_edge_rep = word_rel_pos_edge_rep.cuda(non_blocking=True)
                deprel_edge_rep = deprel_edge_rep.cuda(non_blocking=True)
                deparc_edge_rep = deparc_edge_rep.cuda(non_blocking=True)
                edge_index_rep = edge_index_rep.cuda(non_blocking=True)
                edge_len_rep = edge_len_rep.cuda(non_blocking=True)
                dist_edge_rep = dist_edge_rep.cuda(non_blocking=True)
                deprel_ext_edge_rep = deprel_ext_edge_rep.cuda(non_blocking=True)
                deprel_ext_path_edge_rep = deprel_ext_path_edge_rep.cuda(non_blocking=True)

        # Run forward
        if self.is_trans:
            logits, _ = self.network(
                sent_len_rep=sent_len_rep,
                we_len_rep=we_len_rep,
                pos_rep=pos_rep,
                deprel_rep=deprel_rep,
                pred_ind_rep=pred_ind_rep,
                we_offset_rep=we_offset_rep,
                we_input_id_rep=we_input_id_rep,
                adj_mask_rep=adj_mask_rep,
                dep_abs_position_rep=dep_abs_position_rep,
                word_abs_position_rep=word_abs_position_rep,
                fc_mask_rep=fc_mask_rep,
                pred_adj_mat_rep=pred_adj_mat_rep,
                dep_rel_pos_mat_rep=dep_rel_pos_mat_rep,
                word_rel_pos_mat_rep=word_rel_pos_mat_rep,
                deprel_mat_rep=deprel_mat_rep,
                deparc_mat_rep=deparc_mat_rep,
                path_len_mat_rep=path_len_mat_rep,
                deprel_path_mat_rep=deprel_path_mat_rep,
                deparc_path_mat_rep=deparc_path_mat_rep,
                deprel_ext_mat_rep=deprel_ext_mat_rep,
                deprel_ext_path_mat_rep=deprel_ext_path_mat_rep
            )
        else:
            logits, _ = self.network(
                sent_len_rep=sent_len_rep,
                we_len_rep=we_len_rep,
                pos_rep=pos_rep,
                deprel_rep=deprel_rep,
                pred_ind_rep=pred_ind_rep,
                we_offset_rep=we_offset_rep,
                we_input_id_rep=we_input_id_rep,
                dep_abs_position_rep=dep_abs_position_rep,
                word_abs_position_rep=word_abs_position_rep,
                deprel_path_edge_rep=deprel_path_edge_rep,
                deparc_path_edge_rep=deparc_path_edge_rep,
                path_len_edge_rep=path_len_edge_rep,
                dep_rel_pos_edge_rep=dep_rel_pos_edge_rep,
                word_rel_pos_edge_rep=word_rel_pos_edge_rep,
                deprel_edge_rep=deprel_edge_rep,
                deparc_edge_rep=deparc_edge_rep,
                edge_index_rep=edge_index_rep,
                edge_len_rep=edge_len_rep,
                dist_edge_rep=dist_edge_rep,
                deprel_ext_edge_rep=deprel_ext_edge_rep,
                deprel_ext_path_edge_rep=deprel_ext_path_edge_rep
            )

        batch_size, max_sent_len = logits.size(0), logits.size(1)
        logits = logits.view(batch_size * max_sent_len, -1)
        loss = self.criterion(logits, sem_role_rep)
        # TODO: Check if this can be omitted
        # loss = loss.mean()

        self.optimizer.zero_grad()

        loss.backward()

        if self.args.apply_max_grad_norm:
            clip_grad_norm_(self.get_network().parameters(), self.args.max_grad_norm)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        return loss.item()

    def predict(self, ex):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch examples
            replace_unk: replace `unk` tokens while generating predictions
            src_raw: raw source (passage); required to replace `unk` term
        Output:
            predictions: #batch predicted sequences
        """
        # Eval mode
        self.get_network().gtn.embedding.pretrained_word_emb.eval()
        self.network.eval()

        sent_len_rep = ex['sent_len_rep']
        we_len_rep = ex['we_len_rep']
        pos_rep = ex['pos_rep']
        deprel_rep = ex['deprel_rep']
        pred_ind_rep = ex['pred_ind_rep']
        sem_role_rep = ex['sem_role_rep']
        we_offset_rep = ex['we_offset_rep']
        we_input_id_rep = ex['we_input_id_rep']
        word_abs_position_rep = ex['word_abs_position_rep']
        dep_abs_position_rep = ex['dep_abs_position_rep']

        adj_mask_rep = ex['adj_mask_rep'] if self.is_trans else None
        fc_mask_rep = ex['fc_mask_rep'] if self.is_trans else None
        pred_adj_mat_rep = ex['pred_adj_mat_rep'] if self.is_trans else None
        dep_rel_pos_mat_rep = ex['dep_rel_pos_mat_rep'] if self.is_trans else None
        word_rel_pos_mat_rep = ex['word_rel_pos_mat_rep'] if self.is_trans else None
        deprel_mat_rep = ex['deprel_mat_rep'] if self.is_trans else None
        deparc_mat_rep = ex['deparc_mat_rep'] if self.is_trans else None
        path_len_mat_rep = ex['path_len_mat_rep'] if self.is_trans else None
        deprel_path_mat_rep = ex['deprel_path_mat_rep'] if self.is_trans else None
        deparc_path_mat_rep = ex['deparc_path_mat_rep'] if self.is_trans else None
        deprel_ext_mat_rep = ex['deprel_ext_mat_rep'] if self.is_trans else None
        deprel_ext_path_mat_rep = ex['deprel_ext_path_mat_rep'] if self.is_trans else None

        deprel_path_edge_rep = None if self.is_trans else ex['deprel_path_edge_rep']
        deparc_path_edge_rep = None if self.is_trans else ex['deparc_path_edge_rep']
        deprel_ext_path_edge_rep = None if self.is_trans else ex['deprel_ext_path_edge_rep']
        path_len_edge_rep = None if self.is_trans else ex['path_len_edge_rep']
        dep_rel_pos_edge_rep = None if self.is_trans else ex['dep_rel_pos_edge_rep']
        word_rel_pos_edge_rep = None if self.is_trans else ex['word_rel_pos_edge_rep']
        deprel_edge_rep = None if self.is_trans else ex['deprel_edge_rep']
        deparc_edge_rep = None if self.is_trans else ex['deparc_edge_rep']
        edge_index_rep = None if self.is_trans else ex['edge_index_rep']
        edge_len_rep = None if self.is_trans else ex['edge_len_rep']
        dist_edge_rep = None if self.is_trans else ex['dist_edge_rep']
        deprel_ext_edge_rep = None if self.is_trans else ex['deprel_ext_edge_rep']

        if self.use_cuda:
            sent_len_rep = sent_len_rep.cuda(non_blocking=True)
            we_len_rep = we_len_rep.cuda(non_blocking=True)
            pos_rep = pos_rep.cuda(non_blocking=True)
            deprel_rep = deprel_rep.cuda(non_blocking=True)
            pred_ind_rep = pred_ind_rep.cuda(non_blocking=True)
            sem_role_rep = sem_role_rep.cuda(non_blocking=True)
            we_offset_rep = we_offset_rep.cuda(non_blocking=True)
            we_input_id_rep = we_input_id_rep.cuda(non_blocking=True)
            word_abs_position_rep = word_abs_position_rep.cuda(non_blocking=True)
            dep_abs_position_rep = dep_abs_position_rep.cuda(non_blocking=True)

            if self.is_trans:
                adj_mask_rep = adj_mask_rep.cuda(non_blocking=True)
                fc_mask_rep = fc_mask_rep.cuda(non_blocking=True)
                pred_adj_mat_rep = pred_adj_mat_rep.cuda(non_blocking=True)
                dep_rel_pos_mat_rep = dep_rel_pos_mat_rep.cuda(non_blocking=True)
                word_rel_pos_mat_rep = word_rel_pos_mat_rep.cuda(non_blocking=True)
                deprel_mat_rep = deprel_mat_rep.cuda(non_blocking=True)
                deparc_mat_rep = deparc_mat_rep.cuda(non_blocking=True)
                path_len_mat_rep = path_len_mat_rep.cuda(non_blocking=True)
                deprel_path_mat_rep = deprel_path_mat_rep.cuda(non_blocking=True)
                deparc_path_mat_rep = deparc_path_mat_rep.cuda(non_blocking=True)
                deprel_ext_mat_rep = deprel_ext_mat_rep.cuda(non_blocking=True)
                deprel_ext_path_mat_rep = deprel_ext_path_mat_rep.cuda(non_blocking=True)
            else:
                deprel_path_edge_rep = deprel_path_edge_rep.cuda(non_blocking=True)
                deparc_path_edge_rep = deparc_path_edge_rep.cuda(non_blocking=True)
                path_len_edge_rep = path_len_edge_rep.cuda(non_blocking=True)
                dep_rel_pos_edge_rep = dep_rel_pos_edge_rep.cuda(non_blocking=True)
                word_rel_pos_edge_rep = word_rel_pos_edge_rep.cuda(non_blocking=True)
                deprel_edge_rep = deprel_edge_rep.cuda(non_blocking=True)
                deparc_edge_rep = deparc_edge_rep.cuda(non_blocking=True)
                edge_index_rep = edge_index_rep.cuda(non_blocking=True)
                edge_len_rep = edge_len_rep.cuda(non_blocking=True)
                dist_edge_rep = dist_edge_rep.cuda(non_blocking=True)
                deprel_ext_edge_rep = deprel_ext_edge_rep.cuda(non_blocking=True)
                deprel_ext_path_edge_rep = deprel_ext_path_edge_rep.cuda(non_blocking=True)

        if self.is_trans:
            logits, _ = self.network(
                sent_len_rep=sent_len_rep,
                we_len_rep=we_len_rep,
                pos_rep=pos_rep,
                deprel_rep=deprel_rep,
                pred_ind_rep=pred_ind_rep,
                we_offset_rep=we_offset_rep,
                we_input_id_rep=we_input_id_rep,
                adj_mask_rep=adj_mask_rep,
                word_abs_position_rep=word_abs_position_rep,
                dep_abs_position_rep=dep_abs_position_rep,
                fc_mask_rep=fc_mask_rep,
                pred_adj_mat_rep=pred_adj_mat_rep,
                dep_rel_pos_mat_rep=dep_rel_pos_mat_rep,
                word_rel_pos_mat_rep=word_rel_pos_mat_rep,
                deprel_mat_rep=deprel_mat_rep,
                deparc_mat_rep=deparc_mat_rep,
                path_len_mat_rep=path_len_mat_rep,
                deprel_path_mat_rep=deprel_path_mat_rep,
                deparc_path_mat_rep=deparc_path_mat_rep,
                deprel_ext_mat_rep=deprel_ext_mat_rep,
                deprel_ext_path_mat_rep=deprel_ext_path_mat_rep
            )
        else:
            logits, _ = self.network(
                sent_len_rep=sent_len_rep,
                we_len_rep=we_len_rep,
                pos_rep=pos_rep,
                deprel_rep=deprel_rep,
                pred_ind_rep=pred_ind_rep,
                we_offset_rep=we_offset_rep,
                we_input_id_rep=we_input_id_rep,
                word_abs_position_rep=word_abs_position_rep,
                dep_abs_position_rep=dep_abs_position_rep,
                deprel_path_edge_rep=deprel_path_edge_rep,
                deparc_path_edge_rep=deparc_path_edge_rep,
                path_len_edge_rep=path_len_edge_rep,
                dep_rel_pos_edge_rep=dep_rel_pos_edge_rep,
                word_rel_pos_edge_rep=word_rel_pos_edge_rep,
                deprel_edge_rep=deprel_edge_rep,
                deparc_edge_rep=deparc_edge_rep,
                edge_index_rep=edge_index_rep,
                edge_len_rep=edge_len_rep,
                dist_edge_rep=dist_edge_rep,
                deprel_ext_edge_rep=deprel_ext_edge_rep,
                deprel_ext_path_edge_rep=deprel_ext_path_edge_rep
            )

        batch_size, max_sent_len = logits.size(0), logits.size(1)
        logits = logits.view(batch_size * max_sent_len, -1)
        loss = self.criterion(logits, sem_role_rep)
        probs = f.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()

        return {
            'loss': loss,
            'probs': probs,
            'predictions': predictions
        }

    def save(self, filename):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        params = {
            'state_dict': state_dict,
            'deprel_voc_by_version': self.deprel_voc_by_version,
            'deprel_ext_voc_by_version': self.deprel_ext_voc_by_version,
            'pos_voc': self.pos_voc,
            'semantic_role_voc': self.semantic_role_voc,
            'pred_ind_voc': self.pred_ind_voc,
            'deparc_voc': self.deparc_voc,
            'args': self.args
        }

        torch.save(params, filename)

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        params = {
            'state_dict': state_dict,
            'deprel_voc_by_version': self.deprel_voc_by_version,
            'deprel_ext_voc_by_version': self.deprel_ext_voc_by_version,
            'pos_voc': self.pos_voc,
            'semantic_role_voc': self.semantic_role_voc,
            'pred_ind_voc': self.pred_ind_voc,
            'deparc_voc': self.deparc_voc,
            'args': self.args,
            'epoch': epoch,
            'updates': self.updates,
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(params, filename)

    @staticmethod
    def load(filename, logger):
        logger.info(f'Loading model: {filename}.')
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        args = saved_params['args']
        model = SemanticRoleLabeler(
            args,
            state_dict=saved_params['state_dict'],
            vocab=saved_params
        )
        return model

    @staticmethod
    def load_checkpoint(filename, logger, use_gpu=True):
        logger.info(f'Loading model from: {filename}.')
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        model = SemanticRoleLabeler(
            args=saved_params['args'],
            state_dict=saved_params['state_dict'],
            vocab=saved_params
        )
        model.updates = saved_params['updates']
        model.init_optimizer(saved_params['optimizer'], use_gpu)
        return model, saved_params['epoch']

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()
        self.criterion = self.criterion.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()
        self.criterion = self.criterion.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
