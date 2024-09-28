keys = [
    'upb_version',
    'num_epochs',
    'num_data_workers',
    'log_dir',
    'dataset_dir',
    'model_dir',
    'random_seed',
    'pretrained_we_layer_extraction_mode',
    'pretrained_we_pos_extraction_mode',
    'optimizer',
    'learning_rate',
    'momentum',
    'weight_decay',
    'valid_metric',
    'num_early_stop',
    'num_decay_epoch',
    'lr_decay',
    'min_lr',
    'max_grad_norm',
    'pretrained_we_model_name',
    'fine_tuned_we',
    'we_out_dim',
    'batch_size',
    'pos_dim',
    'deprel_dim',
    'abs_position_dim',
    'use_dep_abs_position',
    'use_word_abs_position',
    'bias',
    'init_weight_xavier_uniform',
    'pred_ind_dim',
    'emb_dropout',
    'enc_dropout',
    'lstm_dropout',
    'hid_dim',
    'num_embed_graph_heads',
    'max_tree_dists',
    'max_relative_positions',
    'dep_max_relative_positions',
    'num_enc_layers',
    'num_heads',
    'use_sent_rep',
    'use_neg_dist',
    'use_dep_rel_pos',
    'use_word_rel_pos',
    'use_deprel',
    'use_deprel_ext',
    'normalized',
    'use_dep_path',
    'sum_dep_path',
    'use_dep_ext_path',
    'use_dep_path_from_pred',
    'num_mlp_layers',
    'd_k',
    'd_v',
    'd_ff',
    'network_type',
    'base_size',
    'use_positional_embedding',
    'deprel_ext_edge_dim',
    'deprel_edge_dim',
    'neighbor_dim',
    'deparc_edge_dim',
    'rel_pos_dim',
    'att_dim',
    'omit_self_edge',
    'proj_self_edge',
    'concat_input',
    'apply_gated_head',
    'use_slanted_triangle_learning',
    'warmup_epochs',
    'cooldown_epochs',
    'num_runs',
    'idx_start_run',
    'draw_conf_matrix',
    'apply_max_grad_norm',
    'gnn_same_relation_dim',
    'gnn_concat_at_final_layer',
    'gnn_activation_at_final_layer',
    'gnn_activation',
    'gnn_dropout_final',
    'gnn_residual',
    'gnn_layer_norm',
    'gnn_final_bias',
    'gnn_average_heads',
    'gnn_fully_connected',
    'lstm_num_layers',
    'lstm_hidden_size',
    'lstm_dropout_net',
    'lstm_activation',
    'lstm_dropout_net_final',
    'lstm_residual',
    'lstm_layer_norm'
]

arg = {
    'upb_version': {
        'v': '2',
        'w': True
    },
    'num_epochs': {
        'v': '100',
        'w': True
    },
    'optimizer': {
        'v': 'sgd',
        'w': True
    },
    'learning_rate': {
        'v': '0.1',
        'w': True
    },
    'num_data_workers': {
        'v': '2',
        'w': True
    },
    'dataset_dir': {
        'v': 'datasets',
        'w': True
    },
    'random_seed': {
        'v': '$SEED',
        'w': True
    },
    'pretrained_we_layer_extraction_mode': {
        'v': 'last_four_cat',
        'w': True
    },
    'pretrained_we_pos_extraction_mode': {
        'v': 'avg',
        'w': True
    },
    'momentum': {
        'v': '0',
        'w': True
    },
    'weight_decay': {
        'v': '0',
        'w': True
    },
    'valid_metric': {
        'v': 'f1',
        'w': True
    },
    'num_early_stop': {
        'v': '20',
        'w': True
    },
    'num_decay_epoch': {
        'v': '5',
        'w': True
    },
    'lr_decay': {
        'v': '0.9',
        'w': True
    },
    'min_lr': {
        'v': '10e-6',
        'w': True
    },
    'max_grad_norm': {
        'v': '5.0',
        'w': True
    },
    'pretrained_we_model_name': {
        'v': 'bert-base-multilingual-cased',
        'w': True
    },
    'fine_tuned_we': {
        'v': '',
        'w': False
    },
    'we_out_dim': {
        'v': '0',
        'w': True
    },
    'batch_size': {
        'v': '32',
        'w': True
    },
    'pos_dim': {
        'v': '30',
        'w': True
    },
    'bias': {
        'v': '',
        'w': True
    },
    'init_weight_xavier_uniform': {
        'v': '',
        'w': True
    },
    'pred_ind_dim': {
        'v': '30',
        'w': True
    },
    'emb_dropout': {
        'v': '0.5',
        'w': True
    },
    'hid_dim': {
        'v': '512',
        'w': True
    },
    'num_heads': {
        'v': '8',
        'w': True
    },
    'use_sent_rep': {
        'v': '',
        'w': True
    },
    'use_dep_path_from_pred': {
        'v': '',
        'w': False
    },
    'num_mlp_layers': {
        'v': '2',
        'w': True
    },
    'd_ff': {
        'v': '2048',
        'w': True
    },
    'use_positional_embedding': {
        'v': '',
        'w': True
    },
    'neighbor_dim': {
        'v': '512',
        'w': True
    },
    'omit_self_edge': {
        'v': '',
        'w': False
    },
    'proj_self_edge': {
        'v': '',
        'w': False
    },
    'concat_input': {
        'v': '',
        'w': False
    },
    'apply_gated_head': {
        'v': '',
        'w': False
    },
    'use_slanted_triangle_learning': {
        'v': '',
        'w': False
    },
    'warmup_epochs': {
        'v': '1',
        'w': True
    },
    'cooldown_epochs': {
        'v': '15',
        'w': True
    },
    'num_runs': {
        'v': '1',
        'w': True
    },
    'idx_start_run': {
        'v': '0',
        'w': True
    },
    'draw_conf_matrix': {
        'v': '',
        'w': False
    },
    'apply_max_grad_norm': {
        'v': '',
        'w': True
    },
    'gnn_same_relation_dim': {
        'v': '',
        'w': False
    },
    'gnn_concat_at_final_layer': {
        'v': '',
        'w': False
    },
    'gnn_activation_at_final_layer': {
        'v': '',
        'w': True
    },
    'gnn_dropout_final': {
        'v': '',
        'w': True
    },
    'gnn_residual': {
        'v': '',
        'w': True
    },
    'gnn_final_bias': {
        'v': '',
        'w': True
    },
    'gnn_average_heads': {
        'v': '',
        'w': False
    },
    'gnn_fully_connected': {
        'v': '',
        'w': False
    },
    'lstm_dropout_net_final': {
        'v': '',
        'w': True
    },
    'lstm_residual': {
        'v': '',
        'w': True
    },
    'dep_max_relative_positions': {
        'v': '0',
        'w': True
    }
}

model = {
    'dummy': {
        'network_type': {
            'v': '',
            'w': True
        },
        'deprel_dim': {
            'v': '',
            'w': True
        },
        'abs_position_dim': {
            'v': '',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': True
        },
        'enc_dropout': {
            'v': '',
            'w': True
        },
        'lstm_dropout': {
            'v': '',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '',
            'w': True
        },
        'max_tree_dists': {
            'v': '',
            'w': True
        },
        'max_relative_positions': {
            'v': '',
            'w': True
        },
        'dep_max_relative_positions': {
            'v': '',
            'w': True
        },
        'num_enc_layers': {
            'v': '',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': True
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': True
        },
        'use_word_rel_pos': {
            'v': '',
            'w': True
        },
        'use_deprel': {
            'v': '',
            'w': True
        },
        'use_deprel_ext': {
            'v': '',
            'w': True
        },
        'normalized': {
            'v': '',
            'w': True
        },
        'use_dep_path': {
            'v': '',
            'w': True
        },
        'sum_dep_path': {
            'v': '',
            'w': True
        },
        'use_dep_ext_path': {
            'v': '',
            'w': True
        },
        'd_k': {
            'v': '',
            'w': True
        },
        'd_v': {
            'v': '',
            'w': True
        },
        'base_size': {
            'v': '',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '',
            'w': True
        },
        'rel_pos_dim': {
            'v': '',
            'w': True
        },
        'att_dim': {
            'v': '',
            'w': True
        },
        'gnn_activation': {
            'v': '',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': True
        },
        'lstm_num_layers': {
            'v': '',
            'w': True
        },
        'lstm_hidden_size': {
            'v': '',
            'w': True
        },
        'lstm_dropout_net': {
            'v': '',
            'w': True
        },
        'lstm_activation': {
            'v': '',
            'w': True
        },
        'lstm_layer_norm': {
            'v': '',
            'w': True
        }
    },

    'gcn_syn': {
        'network_type': {
            'v': 'gcn_syntactic',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '0',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': False
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '1',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': True
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '512',
            'w': True
        },
        'd_v': {
            'v': '512',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '0',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '0',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '0',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'relu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'gnn_activation_at_final_layer': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'gcn_ar': {
        'network_type': {
            'v': 'gcn_ar',
            'w': True
        },
        'enc_dropout': {
            'v': '0.5',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '0',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': False
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '1',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '16',
            'w': True
        },
        'num_enc_layers': {
            'v': '1',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': True
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': True
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '8',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '4',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '4',
            'w': True
        },
        'rel_pos_dim': {
            'v': '64',
            'w': True
        },
        'att_dim': {
            'v': '16',
            'w': True
        },
        'gnn_activation': {
            'v': 'relu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'gnn_activation_at_final_layer': {
            'v': '',
            'w': True
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'gat_plain': {
        'network_type': {
            'v': 'gat_plain',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'enc_dropout': {
            'v': '0.1',
            'w': True
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '1',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'num_enc_layers': {
            'v': '2',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '0',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '0',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '0',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'leaky_relu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'gnn_activation_at_final_layer': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'lstm': {
        'network_type': {
            'v': 'lstm',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'enc_dropout': {
            'v': '0.3',
            'w': True
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '1',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'num_enc_layers': {
            'v': '4',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '512',
            'w': True
        },
        'd_v': {
            'v': '512',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '0',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '0',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '0',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '4',
            'w': True
        },
        'lstm_hidden_size': {
            'v': '256',
            'w': True
        },
        'lstm_dropout_net': {
            'v': '0.3',
            'w': True
        },
        'lstm_activation': {
            'v': 'relu',
            'w': True
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'gat_het': {
        'network_type': {
            'v': 'gat_het',
            'w': True
        },
        'deprel_dim': {
            'v': '0',
            'w': True
        },
        'abs_position_dim': {
            'v': '0',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': False
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'enc_dropout': {
            'v': '0.3',
            'w': True
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '1',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'num_enc_layers': {
            'v': '2',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': True
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '16',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '8',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '8',
            'w': True
        },
        'rel_pos_dim': {
            'v': '16',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'leaky_relu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'gnn_activation_at_final_layer': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'gat_two_att': {
        'network_type': {
            'v': 'gat_two_att',
            'w': True
        },
        'deprel_dim': {
            'v': '0',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'enc_dropout': {
            'v': '0.3',
            'w': True
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '1',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '16',
            'w': True
        },
        'num_enc_layers': {
            'v': '2',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': True
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': True
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '16',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '8',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '8',
            'w': True
        },
        'rel_pos_dim': {
            'v': '128',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'leaky_relu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'gnn_activation_at_final_layer': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'gat_kb': {
        'network_type': {
            'v': 'gat_kb',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '0',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': False
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '1',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '16',
            'w': True
        },
        'num_enc_layers': {
            'v': '2',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': True
        },
        'use_deprel': {
            'v': '',
            'w': True
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '32',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '24',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '8',
            'w': True
        },
        'rel_pos_dim': {
            'v': '32',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'leaky_relu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'gnn_activation_at_final_layer': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'gcn_r': {
        'network_type': {
            'v': 'gcn_relational',
            'w': True
        },
        'enc_dropout': {
            'v': '0.5',
            'w': True
        },
        'deprel_dim': {
            'v': '0',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '1',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'num_enc_layers': {
            'v': '1',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': True
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '512',
            'w': True
        },
        'd_v': {
            'v': '512',
            'w': True
        },
        'base_size': {
            'v': '2',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '0',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '0',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '0',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'relu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'gnn_activation_at_final_layer': {
            'v': '',
            'w': True
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },

    'trans_rpr': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '0',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': False
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '16',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': True
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'gate': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '4',
            'w': True
        },
        'max_tree_dists': {
            'v': '4 4 8 8',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': True
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': False
        },
        'use_word_abs_position': {
            'v': '',
            'w': True
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_spr': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '0',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_spr_rel': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '0',
            'w': True
        },
        'abs_position_dim': {
            'v': '0',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': False
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'dep_max_relative_positions': {
            'v': '2',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': True
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_rpr_x_spr': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '0',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '16',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': True
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_x_spr_rel': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': False
        },
        'use_word_abs_position': {
            'v': '',
            'w': True
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'dep_max_relative_positions': {
            'v': '2',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': True
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_x_spr_rel_x_dr': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': False
        },
        'use_word_abs_position': {
            'v': '',
            'w': True
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'dep_max_relative_positions': {
            'v': '2',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': True
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': True
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '60',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '4',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_rpr_x_spr_x_dr': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '0',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': True
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '60',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '4',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_rpr_x_spr_x_ldp': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '0',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': True
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '62',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '2',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_rpr_x_spr_x_sdp': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '0',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': True
        },
        'sum_dep_path': {
            'v': '',
            'w': True
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '0',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '0',
            'w': True
        },
        'att_dim': {
            'v': '0',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },

    'trans_x_rpr': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': False
        },
        'use_word_abs_position': {
            'v': '',
            'w': True
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '16',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': True
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '16',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '64',
            'w': True
        },
        'att_dim': {
            'v': '8',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_spr_x_spr_rel': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'dep_max_relative_positions': {
            'v': '2',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': True
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '16',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '64',
            'w': True
        },
        'att_dim': {
            'v': '8',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_x_spr': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': True
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '16',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '64',
            'w': True
        },
        'att_dim': {
            'v': '8',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_x_spr_x_spr_rel': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': True
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '0',
            'w': True
        },
        'dep_max_relative_positions': {
            'v': '2',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': True
        },
        'use_word_rel_pos': {
            'v': '',
            'w': False
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '16',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '64',
            'w': True
        },
        'att_dim': {
            'v': '8',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_rpr_x_spr_rel': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '0',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': False
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '16',
            'w': True
        },
        'dep_max_relative_positions': {
            'v': '2',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': True
        },
        'use_word_rel_pos': {
            'v': '',
            'w': True
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '16',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '64',
            'w': True
        },
        'att_dim': {
            'v': '8',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_rpr_x_spr_x_spr_rel': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': False
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '16',
            'w': True
        },
        'dep_max_relative_positions': {
            'v': '2',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': True
        },
        'use_word_rel_pos': {
            'v': '',
            'w': True
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '16',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '64',
            'w': True
        },
        'att_dim': {
            'v': '8',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_x_rpr_x_spr': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': True
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '16',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': False
        },
        'use_word_rel_pos': {
            'v': '',
            'w': True
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '16',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '64',
            'w': True
        },
        'att_dim': {
            'v': '8',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    },
    'trans_x_rpr_x_spr_rel': {
            'network_type': {
                'v': 'trans',
                'w': True
            },
            'enc_dropout': {
                'v': '0.2',
                'w': True
            },
            'deprel_dim': {
                'v': '30',
                'w': True
            },
            'abs_position_dim': {
                'v': '30',
                'w': True
            },
            'use_dep_abs_position': {
                'v': '',
                'w': False
            },
            'use_word_abs_position': {
                'v': '',
                'w': True
            },
            'lstm_dropout': {
                'v': '0',
                'w': True
            },
            'num_embed_graph_heads': {
                'v': '0',
                'w': True
            },
            'max_tree_dists': {
                'v': '1',
                'w': True
            },
            'max_relative_positions': {
                'v': '16',
                'w': True
            },
            'dep_max_relative_positions': {
                'v': '2',
                'w': True
            },
            'num_enc_layers': {
                'v': '3',
                'w': True
            },
            'use_neg_dist': {
                'v': '',
                'w': False
            },
            'use_dep_rel_pos': {
                'v': '',
                'w': True
            },
            'use_word_rel_pos': {
                'v': '',
                'w': True
            },
            'use_deprel': {
                'v': '',
                'w': False
            },
            'use_deprel_ext': {
                'v': '',
                'w': False
            },
            'normalized': {
                'v': '',
                'w': False
            },
            'use_dep_path': {
                'v': '',
                'w': False
            },
            'sum_dep_path': {
                'v': '',
                'w': False
            },
            'use_dep_ext_path': {
                'v': '',
                'w': False
            },
            'd_k': {
                'v': '64',
                'w': True
            },
            'd_v': {
                'v': '64',
                'w': True
            },
            'base_size': {
                'v': '16',
                'w': True
            },
            'deprel_ext_edge_dim': {
                'v': '64',
                'w': True
            },
            'deprel_edge_dim': {
                'v': '32',
                'w': True
            },
            'deparc_edge_dim': {
                'v': '32',
                'w': True
            },
            'rel_pos_dim': {
                'v': '64',
                'w': True
            },
            'att_dim': {
                'v': '8',
                'w': True
            },
            'gnn_activation': {
                'v': 'elu',
                'w': True
            },
            'gnn_layer_norm': {
                'v': '',
                'w': False
            },
            'lstm_num_layers': {
                'v': '',
                'w': False
            },
            'lstm_hidden_size': {
                'v': '',
                'w': False
            },
            'lstm_dropout_net': {
                'v': '',
                'w': False
            },
            'lstm_activation': {
                'v': '',
                'w': False
            },
            'lstm_layer_norm': {
                'v': '',
                'w': False
            }
        },
    'trans_x_rpr_x_spr_x_spr_rel': {
        'network_type': {
            'v': 'trans',
            'w': True
        },
        'enc_dropout': {
            'v': '0.2',
            'w': True
        },
        'deprel_dim': {
            'v': '30',
            'w': True
        },
        'abs_position_dim': {
            'v': '30',
            'w': True
        },
        'use_dep_abs_position': {
            'v': '',
            'w': True
        },
        'use_word_abs_position': {
            'v': '',
            'w': True
        },
        'lstm_dropout': {
            'v': '0',
            'w': True
        },
        'num_embed_graph_heads': {
            'v': '0',
            'w': True
        },
        'max_tree_dists': {
            'v': '1',
            'w': True
        },
        'max_relative_positions': {
            'v': '16',
            'w': True
        },
        'dep_max_relative_positions': {
            'v': '2',
            'w': True
        },
        'num_enc_layers': {
            'v': '3',
            'w': True
        },
        'use_neg_dist': {
            'v': '',
            'w': False
        },
        'use_dep_rel_pos': {
            'v': '',
            'w': True
        },
        'use_word_rel_pos': {
            'v': '',
            'w': True
        },
        'use_deprel': {
            'v': '',
            'w': False
        },
        'use_deprel_ext': {
            'v': '',
            'w': False
        },
        'normalized': {
            'v': '',
            'w': False
        },
        'use_dep_path': {
            'v': '',
            'w': False
        },
        'sum_dep_path': {
            'v': '',
            'w': False
        },
        'use_dep_ext_path': {
            'v': '',
            'w': False
        },
        'd_k': {
            'v': '64',
            'w': True
        },
        'd_v': {
            'v': '64',
            'w': True
        },
        'base_size': {
            'v': '16',
            'w': True
        },
        'deprel_ext_edge_dim': {
            'v': '64',
            'w': True
        },
        'deprel_edge_dim': {
            'v': '32',
            'w': True
        },
        'deparc_edge_dim': {
            'v': '32',
            'w': True
        },
        'rel_pos_dim': {
            'v': '64',
            'w': True
        },
        'att_dim': {
            'v': '8',
            'w': True
        },
        'gnn_activation': {
            'v': 'elu',
            'w': True
        },
        'gnn_layer_norm': {
            'v': '',
            'w': False
        },
        'lstm_num_layers': {
            'v': '',
            'w': False
        },
        'lstm_hidden_size': {
            'v': '',
            'w': False
        },
        'lstm_dropout_net': {
            'v': '',
            'w': False
        },
        'lstm_activation': {
            'v': '',
            'w': False
        },
        'lstm_layer_norm': {
            'v': '',
            'w': False
        }
    }
}


def generate_file(
    file_name,
    spec_arg,
    exp_arg
):
    f = open(f'../scripts/tests/{file_name}.sh', 'w')
    f.write('SEED=1234\n\n')
    f.write('python main.py \\\n')
    for key in keys:
        if key in exp_arg:
            chosen_arg = exp_arg[key]
        elif key in spec_arg:
            chosen_arg = spec_arg[key]
        else:
            chosen_arg = arg[key]

        if chosen_arg['w']:
            if chosen_arg['v'] == '':
                f.write(' '.join([f'--{key}', '\\\n']))
            else:
                f.write(' '.join([f'--{key}', chosen_arg['v'], '\\\n']))
    f.write(' '.join([f'--gold', '\n']))

    f.write('python main.py \\\n')
    for key in keys:
        if key in exp_arg:
            chosen_arg = exp_arg[key]
        elif key in spec_arg:
            chosen_arg = spec_arg[key]
        else:
            chosen_arg = arg[key]

        if chosen_arg['w']:
            if chosen_arg['v'] == '':
                f.write(' '.join([f'--{key}', '\\\n']))
            else:
                f.write(' '.join([f'--{key}', chosen_arg['v'], '\\\n']))
    f.write(' '.join([f'--test', '\\\n']))
    f.write(' '.join([f'--gold', '\n']))

    f.write('python main.py \\\n')
    for key in keys:
        if key in exp_arg:
            chosen_arg = exp_arg[key]
        elif key in spec_arg:
            chosen_arg = spec_arg[key]
        else:
            chosen_arg = arg[key]

        if chosen_arg['w']:
            if chosen_arg['v'] == '':
                f.write(' '.join([f'--{key}', '\\\n']))
            else:
                f.write(' '.join([f'--{key}', chosen_arg['v'], '\\\n']))
    f.write(' '.join([f'--test', '\n']))

    # f.write(f'python read_evaluation_result.py --num_runs {arg["num_runs"]["v"]} --log_dir {exp_arg["log_dir"]["v"]} --upb_version {arg["upb_version"]["v"]}\n')
    # f.write(f'python read_evaluation_result_dep_dist.py --num_runs {arg["num_runs"]["v"]} --log_dir {exp_arg["log_dir"]["v"]} --upb_version {arg["upb_version"]["v"]}\n')
    f.close()


def generate_param(
    param,
    param_keys,
    idx,
    res,
    collection
):
    key = param_keys[idx]
    for value in param[key]:
        if isinstance(value, bool):
            m = {
                key: {
                    'v': '',
                    'w': value
                }
            }
        else:
            m = {
                key: {
                    'v': value,
                    'w': True
                }
            }

        m.update(res)

        if idx < len(param_keys) - 1:
            generate_param(param, param_keys, idx + 1, m, collection)
        else:
            collection.append(m)


if __name__ == '__main__':
    r = '10'

    abbr = {
        'batch_size': 'bs',
        'fine_tuned_we': 'fwe',
        'optimizer': 'opt',
        'learning_rate': 'lr',
        'num_enc_layers': 'e_layer',
        'enc_dropout': 'e_do',
        'max_tree_dists': 'tree_dist',
        'lstm_num_layers': 'l_layer',
        'lstm_dropout_net': 'l_do',
        'max_relative_positions': 'max_rel',
        'gnn_activation': 'gnn_act',
        'gnn_activation_at_final_layer': 'gnn_act_final',
        'gnn_layer_norm': 'gnn_lnorm',
        'lstm_activation': 'lstm_act',
        'lstm_layer_norm': 'lstm_lnorm',
        'deprel_ext_edge_dim': 'dep_ext_dim',
        'rel_pos_dim': 'rel_pos_dim',
        'use_word_rel_pos': 'word_rel',
        'use_dep_rel_pos': 'dep_rel',
        'use_word_abs_position': 'word_abs',
        'use_dep_abs_position': 'dep_abs',
        'att_dim': 'att_dim',
        'lstm_hidden_size': 'lstm_hid',
        'base_size': 'base',
        'deprel_dim': 'dep_dim',
        'abs_position_dim': 'abs_pos_dim',
        'use_deprel': 'dep',
        'use_deprel_ext': 'dep_ext',
        'deprel_edge_dim': 'dep_rel_dim',
        'deparc_edge_dim': 'dep_arc_dim',
        'use_dep_path': 'dep_path',
        'use_dep_ext_path': 'dep_ext_path',
        'sum_dep_path': 'sum_path',
        'idx_start_run': 'st',
        'pretrained_we_model_name': 'we',
        'pred_ind_dim': 'pred_dim',
        'pos_dim': 'pos_dim'
    }

    params = {
        # 'idx_start_run': ['0', '1', '2', '3', '4'],
        # 'abs_position_dim': ['0'],
        # 'fine_tuned_we': [True],
        # 'batch_size': ['64'],
        # 'pos_dim': ['0'],
        # 'pred_ind_dim': ['0'],
        # 'deprel_dim': ['0'],
        # 'pretrained_we_model_name': ['xlm-roberta-base'],
        # 'enc_dropout': ['0.2']
        # 'use_dep_path': [True],
        # 'use_dep_ext_path': [True],
        # 'sum_dep_path': [True],
        # 'deprel_edge_dim': ['63'],
        # 'deparc_edge_dim': ['1']
        # 'deprel_dim': ['0', '30'],
        # 'abs_position_dim': ['0', '30'],
        # 'use_word_abs_position': [False],
        # 'use_dep_abs_position': [True],
        # 'use_word_rel_pos': [True, False],
        # 'use_deprel': [True],
        # 'deprel_edge_dim': ['63'],
        # 'deparc_edge_dim': ['1'],
        # 'use_deprel_ext': [True],
        # 'max_relative_positions': ['0', '16'],
        # 'deprel_edge_dim': ['31'],
        # 'deparc_edge_dim': ['1'],
        # 'use_dep_rel_pos': [True, False],
        # 'deprel_ext_edge_dim': ['64'],
        # 'deprel_edge_dim': ['63'],
        # 'deparc_edge_dim': ['1']
    }

    model_list = [
        'gcn_syn',
        'gcn_ar',
        'gat_plain',
        'lstm',
        'gat_het',
        'gat_two_att',
        'gat_kb',
        'gcn_r',
        'trans_rpr',
        'gate',
        'trans',
        'trans_spr',
        'trans_spr_rel',
        'trans_x_spr_rel_x_dr',
        'trans_rpr_x_spr_x_dr',
        'trans_rpr_x_spr',
        'trans_x_spr_rel',
        'trans_rpr_x_spr_x_ldp',
        'trans_rpr_x_spr_x_sdp'
    ]

    col = []

    generate_param(params, list(params.keys()), 0, {}, col)

    f_script = open(f'../script.txt', 'a')
    f_log = open(f'../log.txt', 'a')

    for model_name in model_list:
        print(model_name)
        for exp in col:
            print(exp)
            unique = []

            for i in exp:
                if exp[i]['v'] == '':
                    if exp[i]['w']:
                        unique.append(f"{abbr[i]}_t".replace(' ', ''))
                    else:
                        unique.append(f"{abbr[i]}_f".replace(' ', ''))
                else:
                    unique.append(f"{abbr[i]}_{exp[i]['v']}".replace(' ', ''))

            file_name = f'{model_name}_{r}_{"_".join(unique)}'
            f_script.write(f'srun -p p --gres=gpu:1 --mem=64GB bash scripts/tests/{file_name}.sh\n')
            f_log.write(f'{file_name}_logs\n')

            new_exp = {
                'model_dir': {
                    'v': f'{file_name}_models',
                    'w': True
                },
                'log_dir': {
                    'v': f'{file_name}_logs',
                    'w': True
                }
            }

            new_exp.update(exp)

            generate_file(
                file_name=file_name,
                spec_arg=model[model_name],
                exp_arg=new_exp
            )

    f_script.write('\n')
    f_log.write('\n')
    f_script.close()
    f_log.close()
