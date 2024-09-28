import numpy as np
import os
import torch
from torch.utils.data import Dataset
from constants import dataset, word
from utils.dataset import load_sentences, load_instances
from utils.model import is_trans, generate_relative_positions_matrix


class UPBDataset(Dataset):
    def __init__(
            self,
            model,
            args,
            set_name,
            lang,
            dataset_dist=None
    ):
        self.model = model
        self.lang = lang
        self.set_name = set_name
        self.sentences_by_treebank = dict()
        self.instances_by_treebank = dict()
        self.sentences = []
        self.instances = []
        self.args = args

        self.is_trans = is_trans(args.network_type)

        assert self.set_name in dataset.set_name

        prefix_name = 'prefix_gold' if (args.gold or self.set_name == 'train') else 'prefix_pred'
        subfolder_name = 'gold' if (args.gold or self.set_name == 'train') else 'pred'

        metadata_by_treebank = dataset.metadata_by_version_to_lang_to_treebank[args.upb_version][lang]

        for treebank in metadata_by_treebank:
            metadata = metadata_by_treebank[treebank]

            identifier = '_'.join([
                str(args.upb_version),
                lang,
                treebank,
                set_name,
                self.model.get_network().gtn.embedding.pretrained_word_emb.model_name
            ])

            dataset_filename = os.path.join(
                args.dataset_dir,
                subfolder_name,
                f'dataset_{identifier}.pkl'
            )

            if os.path.isfile(dataset_filename):
                self.sentences_by_treebank[treebank], self.instances_by_treebank[treebank] = torch.load(
                    dataset_filename, map_location=lambda storage, loc: storage
                )
            else:
                self.sentences_by_treebank[treebank] = load_sentences(
                    file_path=metadata[prefix_name] + metadata[set_name],
                    set_name=set_name,
                    model=self.model,
                    loader_version=metadata['load_ver'],
                    upb_version=args.upb_version,
                    logger=self.args.logger
                )

                self.instances_by_treebank[treebank] = load_instances(
                    sentences=self.sentences_by_treebank[treebank],
                    model=self.model
                )

                torch.save(
                    (self.sentences_by_treebank[treebank], self.instances_by_treebank[treebank]),
                    dataset_filename
                )

            self.sentences.extend(self.sentences_by_treebank[treebank])
            self.instances.extend(self.instances_by_treebank[treebank])

        if dataset_dist:
            if dataset_dist.get(lang) is None:
                dataset_dist[lang] = dict()

            if dataset_dist[lang].get(set_name) is None:
                dataset_dist[lang][set_name] = dict()

            dataset_dist[lang][set_name] = {
                'sentences': len(self.sentences),
                'instances': len(self.instances)
            }

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        instance = self.instances[index]
        pos = torch.LongTensor(instance.sentence.upos_ids)
        deprel = torch.LongTensor(instance.sentence.deprel_ids)
        pred_ind = torch.LongTensor(instance.pred_ind_ids)
        yes_idx = self.model.pred_ind_voc[word.pred_ind_map['yes']]
        idx_pred = instance.pred_ind_ids.index(yes_idx)
        # if self.lang == 'en':
        #     pass
        # else:
        #     assert torch.sum(pred_ind == yes_idx).item() == 1, f'{instance.sentence.sent_id}: {torch.sum(pred_ind == yes_idx).item()}'
        sem_role = torch.LongTensor(instance.sem_role_ids)
        we_offset = torch.LongTensor(instance.sentence.we_offsets)
        # TODO: Should we randomly apply dropout to unknown?
        we_input_id = torch.LongTensor(instance.sentence.we_input_ids)

        root = instance.sentence.root

        dep_abs_position = torch.LongTensor(instance.sentence.depths)
        word_abs_position = torch.arange(len(instance.sentence.we_offsets))

        # self: 0
        adj_mask = torch.from_numpy(instance.sentence.dist_matrix).float()
        head = torch.LongTensor(instance.sentence.heads)

        deprel_mat = torch.from_numpy(instance.sentence.deprel_matrix).long()
        deprel_ext_mat = torch.from_numpy(instance.sentence.deprel_ext_matrix).long()
        deparc_mat = torch.from_numpy(instance.sentence.deparc_matrix).long()
        deprel_path_mat = torch.from_numpy(instance.sentence.deprel_path_matrix).long()
        deparc_path_mat = torch.from_numpy(instance.sentence.deparc_path_matrix).long()
        deprel_ext_path_mat = torch.from_numpy(instance.sentence.deprel_ext_path_matrix).long()
        path_len_mat = torch.from_numpy(instance.sentence.path_len_matrix).long()
        pred_adj_mat = torch.from_numpy(instance.pred_adj_matrix).long()
        # assert np.array_equal(
        #     instance.sentence.path_len_matrix[idx_pred],
        #     instance.sentence.path_len_matrix[:, idx_pred]
        # ), instance.sentence.path_len_matrix
        pred_dep_dist = torch.from_numpy(instance.sentence.path_len_matrix[idx_pred]).long()
        pred_dep_dist = torch.empty_like(pred_dep_dist).copy_(pred_dep_dist)
        pred_dep_dist[idx_pred] = 0

        # print('words', instance.sentence.words)
        # print('pos', [instance.sentence.model.pos_voc[p.item()] for p in pos])
        # print('deprel', [instance.sentence.model.deprel_voc_by_version[instance.sentence.upb_version][d.item()] for d in deprel])
        # print('pred_ind', pred_ind)
        # print('sem_role', [instance.sentence.model.semantic_role_voc[s.item()] for s in sem_role])
        # print('we_offset', we_offset)
        # print('we_input_id', we_input_id)
        # print('we_words', instance.sentence.we_words)
        # print('abs_position', abs_position)
        # print('adj_mask', adj_mask)
        # print('head', head)

        return {
            'pos': pos,
            'deprel': deprel,
            'pred_ind': pred_ind,
            'sem_role': sem_role,
            'we_offset': we_offset,
            'we_input_id': we_input_id,
            'root': root,
            'dep_abs_position': dep_abs_position,
            'word_abs_position': word_abs_position,
            'adj_mask': adj_mask,
            'pred_adj_mat': pred_adj_mat,
            'head': head,
            'deprel_mat': deprel_mat,
            'deprel_ext_mat': deprel_ext_mat,
            'deparc_mat': deparc_mat,
            'deprel_path_mat': deprel_path_mat,
            'deparc_path_mat': deparc_path_mat,
            'path_len_mat': path_len_mat,
            'deprel_ext_path_mat': deprel_ext_path_mat,
            'pred_dep_dist': pred_dep_dist
        }

    def batchify(self, batch):
        if not self.is_trans:
            assert ((self.args.num_embed_graph_heads == 1 and len(self.args.max_tree_dists) == 1) or
                    self.args.gnn_fully_connected or
                    self.args.use_dep_path_from_pred)

        # For constructing edges in GNN
        max_dist = self.args.max_tree_dists[0]

        pretrained_word_emb = self.model.get_network().gtn.embedding.pretrained_word_emb
        batch_size = len(batch)
        sent_len = [len(instance['we_offset']) for instance in batch]
        max_sent_len = max(sent_len)
        we_len = [len(instance['we_input_id']) for instance in batch]
        max_we_len = max(we_len)
        path_len = [instance['deprel_path_mat'].shape[2] for instance in batch]
        max_path_len = max(path_len)

        sent_len_rep = torch.LongTensor(sent_len)
        we_len_rep = torch.LongTensor(we_len)

        head_rep = torch.LongTensor(batch_size, max_sent_len).fill_(word.PAD)
        pos_rep = torch.LongTensor(batch_size, max_sent_len).fill_(word.PAD)
        deprel_rep = torch.LongTensor(batch_size, max_sent_len).fill_(word.PAD)
        pred_ind_rep = torch.LongTensor(batch_size, max_sent_len).fill_(word.PAD)
        sem_role_rep = torch.LongTensor(batch_size, max_sent_len).fill_(word.PAD)
        we_offset_rep = torch.LongTensor(batch_size, max_sent_len).fill_(word.PAD)
        we_input_id_rep = torch.LongTensor(batch_size, max_we_len).fill_(
            pretrained_word_emb.padding_id
        )
        pred_dep_dist_rep = torch.LongTensor(batch_size, max_sent_len).zero_()
        adj_mask_rep = torch.FloatTensor(batch_size, max_sent_len, max_sent_len).zero_()
        pred_adj_mat_rep = torch.LongTensor(batch_size, max_sent_len, max_sent_len).zero_()
        fc_mask_rep = torch.FloatTensor(batch_size, max_sent_len, max_sent_len).zero_()
        word_abs_position_rep = torch.LongTensor(batch_size, max_sent_len).fill_(word.DUMMY_MAX_LEN)
        dep_abs_position_rep = torch.LongTensor(batch_size, max_sent_len).fill_(word.DUMMY_MAX_LEN)
        deprel_mat_rep = torch.LongTensor(batch_size, max_sent_len, max_sent_len).fill_(word.PAD)
        deprel_ext_mat_rep = torch.LongTensor(batch_size, max_sent_len, max_sent_len).fill_(word.PAD)
        deparc_mat_rep = torch.LongTensor(batch_size, max_sent_len, max_sent_len).fill_(word.PAD)
        path_len_mat_rep = torch.LongTensor(batch_size, max_sent_len, max_sent_len).zero_()
        deprel_path_mat_rep = torch.LongTensor(batch_size, max_sent_len, max_sent_len, max_path_len).fill_(word.PAD)
        deparc_path_mat_rep = torch.LongTensor(batch_size, max_sent_len, max_sent_len, max_path_len).fill_(word.PAD)
        deprel_ext_path_mat_rep = torch.LongTensor(batch_size, max_sent_len, max_sent_len, max_path_len).fill_(word.PAD)

        dep_rel_pos_mat_rep = torch.LongTensor(batch_size, max_sent_len, max_sent_len).fill_(word.DUMMY_MAX_LEN)
        word_rel_pos_mat_rep = torch.LongTensor(batch_size, max_sent_len, max_sent_len).fill_(word.DUMMY_MAX_LEN)
        template_mat = torch.zeros((max_sent_len, max_sent_len))
        [rows, cols] = torch.triu_indices(max_sent_len, max_sent_len)
        template_mat[rows, cols] = -1
        [rows, cols] = torch.tril_indices(max_sent_len, max_sent_len)
        template_mat[rows, cols] = 1
        template_mat.fill_diagonal_(0)

        for i, instance in enumerate(batch):
            actual_len = sent_len[i]
            if self.args.use_dep_path_from_pred:
                instance['edge_pair'] = torch.nonzero(instance['pred_adj_mat'] == 1)
            elif self.args.gnn_fully_connected:
                instance['edge_pair'] = torch.nonzero(torch.ones(actual_len, actual_len))
            else:
                instance['edge_pair'] = torch.nonzero(torch.logical_and(
                    instance['adj_mask'] >= 1,
                    instance['adj_mask'] <= max_dist
                ))

        edge_len = [len(instance['edge_pair']) for instance in batch]
        max_edge_len = max(edge_len)

        edge_len_rep = torch.LongTensor(edge_len)
        deprel_path_edge_rep = torch.LongTensor(batch_size, max_edge_len, max_path_len).fill_(word.PAD)
        deparc_path_edge_rep = torch.LongTensor(batch_size, max_edge_len, max_path_len).fill_(word.PAD)
        deprel_ext_path_edge_rep = torch.LongTensor(batch_size, max_edge_len, max_path_len).fill_(word.PAD)
        path_len_edge_rep = torch.LongTensor(batch_size, max_edge_len).zero_()
        dep_rel_pos_edge_rep = torch.LongTensor(batch_size, max_edge_len).fill_(word.DUMMY_MAX_LEN)
        word_rel_pos_edge_rep = torch.LongTensor(batch_size, max_edge_len).fill_(word.DUMMY_MAX_LEN)
        deprel_edge_rep = torch.LongTensor(batch_size, max_edge_len).fill_(word.PAD)
        deprel_ext_edge_rep = torch.LongTensor(batch_size, max_edge_len).fill_(word.PAD)
        deparc_edge_rep = torch.LongTensor(batch_size, max_edge_len).fill_(word.PAD)
        edge_index_rep = torch.LongTensor(batch_size, max_edge_len, 2).zero_()
        dist_edge_rep = torch.FloatTensor(batch_size, max_edge_len).zero_()

        for i, instance in enumerate(batch):
            actual_len = sent_len[i]
            actual_we_len = we_len[i]
            actual_path_len = path_len[i]
            actual_edge_len = edge_len[i]

            pred_dep_dist_rep[i, :actual_len] = instance['pred_dep_dist']
            head_rep[i, :actual_len] = instance['head']
            pos_rep[i, :actual_len] = instance['pos']
            deprel_rep[i, :actual_len] = instance['deprel']
            pred_ind_rep[i, :actual_len] = instance['pred_ind']
            sem_role_rep[i, :actual_len] = instance['sem_role']
            we_offset_rep[i, :actual_len] = instance['we_offset']
            we_input_id_rep[i, :actual_we_len] = instance['we_input_id']
            adj_mask_rep[i, :actual_len, :actual_len] = instance['adj_mask']
            pred_adj_mat_rep[i, :actual_len, :actual_len] = instance['pred_adj_mat']
            deprel_mat_rep[i, :actual_len, :actual_len] = instance['deprel_mat']
            deprel_ext_mat_rep[i, :actual_len, :actual_len] = instance['deprel_ext_mat']
            deparc_mat_rep[i, :actual_len, :actual_len] = instance['deparc_mat']
            path_len_mat_rep[i, :actual_len, :actual_len] = instance['path_len_mat']
            deprel_path_mat_rep[i, :actual_len, :actual_len, :actual_path_len] = instance['deprel_path_mat']
            deparc_path_mat_rep[i, :actual_len, :actual_len, :actual_path_len] = instance['deparc_path_mat']
            deprel_ext_path_mat_rep[i, :actual_len, :actual_len, :actual_path_len] = instance['deprel_ext_path_mat']
            fc_mask_rep[i, :actual_len, :actual_len] = torch.ones(actual_len, actual_len)
            word_abs_position_rep[i, :actual_len] = instance['word_abs_position']
            dep_abs_position_rep[i, :actual_len] = instance['dep_abs_position']
            dep_rel_pos_mat_rep[i, :actual_len, :actual_len] = torch.clamp(
                instance['adj_mask'] * template_mat[:actual_len, :actual_len],
                # TODO: Change implementation if we plan to use different max position for each layer
                min=-self.args.max_relative_positions[0] if self.args.dep_max_relative_positions[0] == 0 else -self.args.dep_max_relative_positions[0],
                max=self.args.max_relative_positions[0] if self.args.dep_max_relative_positions[0] == 0 else self.args.dep_max_relative_positions[0]
            ).long()
            word_rel_pos_mat_rep[i, :actual_len, :actual_len] = torch.clamp(
                generate_relative_positions_matrix(actual_len),
                # TODO: Change implementation if we plan to use different max position for each layer
                min=-self.args.max_relative_positions[0],
                max=self.args.max_relative_positions[0]
            ).long()

            x, y = instance['edge_pair'].select(1, 0), instance['edge_pair'].select(1, 1)
            deprel_path_edge_rep[i, :actual_edge_len, :] = deprel_path_mat_rep[i][x, y]
            deparc_path_edge_rep[i, :actual_edge_len, :] = deparc_path_mat_rep[i][x, y]
            deprel_ext_path_edge_rep[i, :actual_edge_len, :] = deprel_ext_path_mat_rep[i][x, y]
            path_len_edge_rep[i, :actual_edge_len] = path_len_mat_rep[i][x, y]
            dep_rel_pos_edge_rep[i, :actual_edge_len] = dep_rel_pos_mat_rep[i][x, y]
            word_rel_pos_edge_rep[i, :actual_edge_len] = word_rel_pos_mat_rep[i][x, y]
            deprel_edge_rep[i, :actual_edge_len] = deprel_mat_rep[i][x, y]
            deprel_ext_edge_rep[i, :actual_edge_len] = deprel_ext_mat_rep[i][x, y]
            deparc_edge_rep[i, :actual_edge_len] = deparc_mat_rep[i][x, y]
            dist_edge_rep[i, :actual_edge_len] = adj_mask_rep[i][x, y]
            edge_index_rep[i, :actual_edge_len, 0] = y
            edge_index_rep[i, :actual_edge_len, 1] = x

        return {
            'head_rep': head_rep,
            'sent_len_rep': sent_len_rep,
            'we_len_rep': we_len_rep,
            'pos_rep': pos_rep,
            'deprel_rep': deprel_rep,
            'pred_ind_rep': pred_ind_rep,
            'sem_role_rep': sem_role_rep.view(batch_size * max_sent_len),
            'we_offset_rep': we_offset_rep,
            'we_input_id_rep': we_input_id_rep,
            'adj_mask_rep': adj_mask_rep,
            'pred_adj_mat_rep': pred_adj_mat_rep,
            'word_abs_position_rep': word_abs_position_rep,
            'dep_abs_position_rep': dep_abs_position_rep,
            'dep_rel_pos_mat_rep': dep_rel_pos_mat_rep,
            'word_rel_pos_mat_rep': word_rel_pos_mat_rep,
            'fc_mask_rep': fc_mask_rep,
            'batch_size': batch_size,
            'deprel_mat_rep': deprel_mat_rep,
            'deprel_ext_mat_rep': deprel_ext_mat_rep,
            'deparc_mat_rep': deparc_mat_rep,
            'path_len_mat_rep': path_len_mat_rep,
            'deprel_path_mat_rep': deprel_path_mat_rep,
            'deparc_path_mat_rep': deparc_path_mat_rep,
            'deprel_ext_path_mat_rep': deprel_ext_path_mat_rep,
            'deprel_path_edge_rep': deprel_path_edge_rep,
            'deparc_path_edge_rep': deparc_path_edge_rep,
            'deprel_ext_path_edge_rep': deprel_ext_path_edge_rep,
            'path_len_edge_rep': path_len_edge_rep,
            'dep_rel_pos_edge_rep': dep_rel_pos_edge_rep,
            'word_rel_pos_edge_rep': word_rel_pos_edge_rep,
            'deprel_edge_rep': deprel_edge_rep,
            'deprel_ext_edge_rep': deprel_ext_edge_rep,
            'deparc_edge_rep': deparc_edge_rep,
            'edge_index_rep': edge_index_rep,
            'edge_len_rep': edge_len_rep,
            'dist_edge_rep': dist_edge_rep,
            'pred_dep_dist_rep': pred_dep_dist_rep.view(batch_size * max_sent_len)
        }
