gat_type = {
    'gat_plain': 'gat_plain',
    'gat_het': 'gat_het',
    'gat_two_att': 'gat_two_att',
    'gat_two_att_ori': 'gat_two_att_ori',
    'gat_kb': 'gat_kb',
    'gaan': 'gaan',
    'san': 'san'
}

gcn_type = {
    'gcn_syntactic': 'gcn_syntactic',
    'gcn_relational': 'gcn_relational',
    'gcn_ar': 'gcn_ar'
}

network_type = {
    'trans': 'trans',
    'trans_gaan': 'trans_gaan',
    'lstm': 'lstm'
}

network_type.update(gat_type)
network_type.update(gcn_type)
