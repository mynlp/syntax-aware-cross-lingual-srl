import argparse


MODEL_ARGS = {
    'upb_version',
    'pretrained_we_layer_extraction_mode',
    'pretrained_we_pos_extraction_mode',
    'pretrained_we_model_name',
    'fine_tuned_we',
    'batch_size',
    'optimizer',
    'learning_rate',
    'momentum',
    'weight_decay',
    'max_grad_norm',
    'emb_dropout',
    'enc_dropout',
    'lstm_dropout',
    'hid_dim',
    'num_enc_layers',
    'num_heads',
    'num_mlp_layers',
    'init_weight_xavier_uniform',
    'we_out_dim',
    'pos_dim',
    'pred_ind_dim',
    'abs_position_dim',
    'deprel_dim',
    'rel_pos_dim',
    'att_dim',
    'base_size',
    'deprel_ext_edge_dim',
    'deprel_edge_dim',
    'deparc_edge_dim',
    'neighbor_dim',
    'd_k',
    'd_v',
    'd_ff',
    'num_embed_graph_heads',
    'max_tree_dists',
    'max_relative_positions',
    'dep_max_relative_positions',
    'use_dep_rel_pos',
    'use_word_rel_pos',
    'use_dep_abs_position',
    'use_word_abs_position',
    'use_neg_dist',
    'bias',
    'network_type',
    'use_positional_embedding',
    'normalized',
    'use_sent_rep',
    'use_deprel',
    'use_deprel_ext',
    'use_dep_path',
    'use_dep_ext_path',
    'use_dep_path_from_pred',
    'sum_dep_path',
    'omit_self_edge',
    'proj_self_edge',
    'concat_input',
    'apply_gated_head',
    'gnn_activation',
    'gnn_activation_at_final_layer',
    'gnn_concat_at_final_layer',
    'gnn_dropout_final',
    'gnn_final_bias',
    'gnn_residual',
    'gnn_layer_norm',
    'gnn_fully_connected',
    'gnn_average_heads',
    'apply_max_grad_norm',
    'gnn_same_relation_dim',
    'lstm_num_layers',
    'lstm_hidden_size',
    'lstm_dropout_net',
    'lstm_activation',
    'lstm_dropout_net_final',
    'lstm_residual',
    'lstm_layer_norm'
}


def get_model_args(args):
    """Filter args for model ones.
    From a args Namespace, return a new Namespace with *only* the args specific
    to the model architecture or optimization. (i.e. the ones defined here.)
    """
    required_args = MODEL_ARGS

    arg_values = {
        k: v for k, v in vars(args).items() if k in required_args
    }
    return argparse.Namespace(**arg_values)


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


def add_arguments(parser):
    parser.add_argument(
        '--upb_version',
        type=int,
        help='UPB version.',
        required=True
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        help='Number of epochs.',
        required=True
    )
    parser.add_argument(
        '--num_data_workers',
        type=int,
        help='Number of subprocesses for data loading.',
        default=5
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        help='Logging directory.',
        required=True
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        help='Dataset directory.',
        default='datasets'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        help='Saved model directory.',
        required=True
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        help='Random seed to avoid randomness.',
        required=True
    )
    parser.add_argument(
        '--gold',
        default=False,
        help='Using gold pos tags and dependency trees.',
        action='store_true'
    )
    parser.add_argument(
        '--test',
        default=False,
        help='Run the evaluation only.',
        action='store_true'
    )
    parser.add_argument(
        '--pretrained_we_layer_extraction_mode',
        type=str,
        help='Layer extraction mode for pretrained word embedding.',
        required=True,
        choices=['last', 'last_four_cat']
    )
    parser.add_argument(
        '--pretrained_we_pos_extraction_mode',
        type=str,
        help='Position extraction mode for pretrained word embedding.',
        required=True,
        choices=['left', 'right', 'avg']
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        help='Optimizer.',
        required=True,
        choices=['sgd', 'adam', 'adamw']
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='Learning rate for the optimizer.',
        required=True
    )
    parser.add_argument(
        '--momentum',
        type=float,
        help='Momentum factor.',
        required=True
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        help='Weight decay factor.',
        required=True
    )
    parser.add_argument(
        '--valid_metric',
        type=str,
        help='The evaluation metric used for model selection.',
        required=True
    )
    parser.add_argument(
        '--num_early_stop',
        type=int,
        help='Stop training if performance does not improve.',
        required=True
    )
    parser.add_argument(
        '--num_decay_epoch',
        type=int,
        help='Decay learning rate after this epoch.',
        required=True
    )
    parser.add_argument(
        '--lr_decay',
        type=float,
        help='Decay ratio for learning rate.',
        required=True
    )
    parser.add_argument(
        '--min_lr',
        type=float,
        help='Minimum allowed learning rate for the optimizer.',
        required=True
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        help='Gradient clipping.',
        required=True
    )
    parser.add_argument(
        '--pretrained_we_model_name',
        type=str,
        help='Pretrained word embedding model name.',
        required=True,
        choices=['xlm-roberta-base', 'bert-base-multilingual-cased']
    )
    parser.add_argument(
        '--fine_tuned_we',
        help='Fine tune pretrained word embedding.',
        default=False,
        action='store_true'
    ),
    parser.add_argument(
        '--we_out_dim',
        type=int,
        help='Force out dim of pretrained word embedding using transformation.',
        default=0
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size.',
        required=True
    )
    parser.add_argument(
        '--pos_dim',
        type=int,
        help='Pos tag embedding dimension.',
        required=True
    )
    parser.add_argument(
        '--deprel_dim',
        type=int,
        help='Deprel embedding dimension.',
        required=True
    )
    parser.add_argument(
        '--abs_position_dim',
        type=int,
        help='Absolute position dimension.',
        required=True
    )
    parser.add_argument(
        '--use_dep_abs_position',
        help='Using dependency-based absolute position.',
        default=False,
        action='store_true'
    ),
    parser.add_argument(
        '--use_word_abs_position',
        help='Using word absolute position.',
        default=False,
        action='store_true'
    ),
    parser.add_argument(
        '--bias',
        help='Use bias for linear transformation in the main network.',
        default=False,
        action='store_true'
    ),
    parser.add_argument(
        '--init_weight_xavier_uniform',
        help='Init weight used for linear transformation in the main network using xavier uniform.',
        default=False,
        action='store_true'
    ),
    parser.add_argument(
        '--pred_ind_dim',
        type=int,
        help='Predicate indicator dimension.',
        required=True
    )
    parser.add_argument(
        '--emb_dropout',
        type=float,
        help='Dropout applied to concatenation of embeddings.',
        required=True
    )
    parser.add_argument(
        '--enc_dropout',
        type=float,
        help='Dropout for encoder.',
        required=True
    )
    parser.add_argument(
        '--lstm_dropout',
        type=float,
        help='Dropout for LSTMs.',
        required=True
    )
    parser.add_argument(
        '--hid_dim',
        type=int,
        help='Model hidden dimension.',
        required=True
    )
    parser.add_argument(
        '--num_embed_graph_heads',
        type=int,
        help='Number of heads to embed dependency graph.',
        required=True
    )
    parser.add_argument(
        '--max_tree_dists',
        nargs='+',
        type=int,
        help='Maximum distance to consider while constructing the adjacency matrix.',
        required=True
    )
    parser.add_argument(
        '--max_relative_positions',
        nargs='+',
        type=int,
        help='Max value for relative position representations.',
        required=True
    )
    parser.add_argument(
        '--dep_max_relative_positions',
        nargs='+',
        type=int,
        help='Max value for dependency relative position representations.',
        default=[0]
    )
    parser.add_argument(
        '--num_enc_layers',
        type=int,
        help='Number of layers in encoder.',
        required=True
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        help='Number of heads in transformers.',
        required=True
    )
    parser.add_argument(
        '--use_sent_rep',
        help='Use sentence representation as features for classifier.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--use_neg_dist',
        help='Use negative value for relative position representations.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--use_dep_rel_pos',
        help='Use dependency-based relative position as relative position representations.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--use_word_rel_pos',
        help='Use word relative position as relative position representations.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--use_deprel',
        help='Use dependency relations concatenated with dependency arcs as edge features.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--use_deprel_ext',
        help='Use dependency relations (extended version) as edge features.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--normalized',
        help='Normalize attention matrix in transformers based on distances.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--use_dep_path',
        help='Use dependency paths as edge features.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--sum_dep_path',
        help='Sum dependency paths as edge features.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--use_dep_ext_path',
        help='Use dependency paths (extended version) as edge features.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--use_dep_path_from_pred',
        help='Use path from predicate to each token to build graph.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--num_mlp_layers',
        type=int,
        help='Number of MLP layers.',
        required=True
    )
    parser.add_argument(
        '--d_k',
        type=int,
        help='Hidden size of heads in multi-head attention.',
        required=True
    )
    parser.add_argument(
        '--d_v',
        type=int,
        help='Hidden size of heads in multi-head attention.',
        required=True
    )
    parser.add_argument(
        '--d_ff',
        type=int,
        help='Number of units in position-wise FFNN.',
        required=True
    )
    parser.add_argument(
        '--gnn_concat_at_final_layer',
        help='GNN concat representation from each head at the final layer.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--gnn_activation_at_final_layer',
        help='GNN apply activation at the final layer.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--gnn_activation',
        type=str,
        help='GNN activation function.',
        required=True,
        choices=['elu', 'relu', 'leaky_relu']
    )
    parser.add_argument(
        '--network_type',
        nargs='+',
        type=str,
        help='Type of networks.',
        required=True
    )
    parser.add_argument(
        '--gnn_dropout_final',
        help='GNN apply dropout at final, before residual connection (similar to transformers).',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--gnn_residual',
        help='Use residual connection at the end of each layer.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--gnn_layer_norm',
        help='Use layer normalization at the end of each layer.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--gnn_final_bias',
        help='GNN apply bias at the final step.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--gnn_average_heads',
        help='GNN average all heads.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--base_size',
        type=int,
        help='Base size to scale down deprel size in relational GCN (avoid overfitting).',
        required=True
    )
    parser.add_argument(
        '--use_positional_embedding',
        help='Use positional embedding (Dai et al., 2019) for relative positions instead of standard embedding.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--deprel_ext_edge_dim',
        type=int,
        help='Dimension for encoding deprel ext as edge properties in ARGCN.',
        required=True
    )
    parser.add_argument(
        '--deprel_edge_dim',
        type=int,
        help='Dimension for encoding deprel as edge properties.',
        required=True
    )
    parser.add_argument(
        '--neighbor_dim',
        type=int,
        help='Dimension for neighbor feed-forward neural network in GaAN.',
        required=True
    )
    parser.add_argument(
        '--deparc_edge_dim',
        type=int,
        help='Dimension for encoding deparc as edge properties.',
        required=True
    )
    parser.add_argument(
        '--rel_pos_dim',
        type=int,
        help='Dimension for encoding relative positions as edge properties.',
        required=True
    )
    parser.add_argument(
        '--att_dim',
        type=int,
        help='Dimension for attention (beta) in ARGCN.',
        required=True
    )
    parser.add_argument(
        '--omit_self_edge',
        help='Whether to omit self edges from neighbors.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--proj_self_edge',
        help='Whether to add projected self edges to final output.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--concat_input',
        help='Whether to concatenate the input before final projection.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--apply_gated_head',
        help='Whether to apply gates when concatenating each head.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--gnn_fully_connected',
        help='Whether to fully connect all words.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--use_slanted_triangle_learning',
        help='Whether to use slanted triangle learning.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--warmup_epochs',
        type=int,
        help='Warmup epochs.',
        required=True
    )
    parser.add_argument(
        '--cooldown_epochs',
        type=int,
        help='Cooldown epochs.',
        required=True
    )
    parser.add_argument(
        '--num_runs',
        type=int,
        help='Number of runs.',
        required=True
    )
    parser.add_argument(
        '--idx_start_run',
        type=int,
        help='Index for start running.',
        default=0
    )
    parser.add_argument(
        '--draw_conf_matrix',
        help='Whether to draw confusion matrix.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--apply_max_grad_norm',
        help='Whether to apply maximum gradient normalization.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--gnn_same_relation_dim',
        help='Use same relation dim in heterogeneous GAT and two-attention GAT.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--lstm_num_layers',
        type=int,
        help='LSTM num layers.',
        required=False
    )
    parser.add_argument(
        '--lstm_hidden_size',
        type=int,
        help='LSTM hidden size.',
        required=False
    )
    parser.add_argument(
        '--lstm_dropout_net',
        type=float,
        help='LSTM dropout.',
        required=False
    )
    parser.add_argument(
        '--lstm_activation',
        type=str,
        help='LSTM activation function.',
        required=False,
        choices=['elu', 'relu', 'leaky_relu', '']
    )
    parser.add_argument(
        '--lstm_dropout_net_final',
        help='LSTM dropout final.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--lstm_residual',
        help='LSTM residual.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--lstm_layer_norm',
        help='LSTM layer norm.',
        default=False,
        action='store_true'
    )
