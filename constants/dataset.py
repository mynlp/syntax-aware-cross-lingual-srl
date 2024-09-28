set_name = {
    'train': 'train',
    'dev': 'dev',
    'test': 'test'
}

lang = {
    'en': 'en',
    'de': 'de',
    'es': 'es',
    'fi': 'fi',
    'fr': 'fr',
    'it': 'it',
    'pt': 'pt',
    'zh': 'zh',
    'cs': 'cs',
    'el': 'el',
    'ko': 'ko',
    'ro': 'ro',
    'hi': 'hi',
    'mr': 'mr',
    'ta': 'ta',
    'hu': 'hu',
    'pl': 'pl',
    'te': 'te',
    'nl': 'nl',
    'id': 'id',
    'ja': 'ja',
    'ru': 'ru',
    'uk': 'uk',
    'vi': 'vi'
}

mandatory_attributes_by_loader_version = {
    1: {
        'token_id': 0,
        'form': 1,
        'lemma': 2,
        'upos': 3,
        'xpos': 4,
        'feats': 5,
        'head': 6,
        'deprel': 7,
        'is_predicate': 8,
        'predicate_sense': 9
    },
    2: {
        'token_id': 0,
        'form': 1,
        'lemma': 2,
        'upos': 3,
        'xpos': 4,
        'feats': 5,
        'head': 6,
        'deprel': 7,
        'deps': 8,
        'misc': 9,
        'predicate_sense': 10
    },
    3: {
        'token_id': 0,
        'form': 1,
        'lemma': 2,
        'upos': 3,
        'xpos': 4,
        'feats': 5,
        'head': 6,
        'deprel': 7,
        'predicate_sense': 8,
        'argument_head': 9,
        'argument_span': 10
    }
}

semantic_role_mapper = {
    'A0': 'ARG0',
    'R-A0': 'R-ARG0',
    'C-A0': 'C-ARG0',
    'A1': 'ARG1',
    'C-A1': 'C-ARG1',
    'R-A1': 'R-ARG1',
    'C-A1-DSP': 'C-ARG1-DSP',
    'A1-DSP': 'ARG1-DSP',
    'A2': 'ARG2',
    'R-A2': 'R-ARG2',
    'C-A2': 'C-ARG2',
    'A3': 'ARG3',
    'R-A3': 'R-ARG3',
    'C-A3': 'C-ARG3',
    'A4': 'ARG4',
    'R-A4': 'R-ARG4',
    'C-A4': 'C-ARG4',
    'A5': 'ARG5',
    'C-AM-DIR': 'C-ARGM-DIR',
    'AM-REC': 'ARGM-REC',
    'R-AM-MNR': 'R-ARGM-MNR',
    'AM-DIS': 'ARGM-DIS',
    'R-AM-TMP': 'R-ARGM-TMP',
    'AM-ADV': 'ARGM-ADV',
    'AM-EXT': 'ARGM-EXT',
    'C-AM-EXT': 'C-ARGM-EXT',
    'R-AM-LOC': 'R-ARGM-LOC',
    'C-AM-LOC': 'C-ARGM-LOC',
    'C-AM-TMP': 'C-ARGM-TMP',
    'AM-PRR': 'ARGM-PRR',
    'C-AM-PRR': 'C-ARGM-PRR',
    'AM-LOC': 'ARGM-LOC',
    'AM-CAU': 'ARGM-CAU',
    'R-AM-CAU': 'R-ARGM-CAU',
    'C-AM-GOL': 'C-ARGM-GOL',
    'AM-COM': 'ARGM-COM',
    'AM-ADJ': 'ARGM-ADJ',
    'R-AM-ADJ': 'R-ARGM-ADJ',
    'AM-NEG': 'ARGM-NEG',
    'AM-PRP': 'ARGM-PRP',
    'C-AM-PRP': 'C-ARGM-PRP',
    'AM-PRD': 'ARGM-PRD',
    'AM-DIR': 'ARGM-DIR',
    'AM-TMP': 'ARGM-TMP',
    'AM-GOL': 'ARGM-GOL',
    'R-AM-COM': 'R-ARGM-COM',
    'C-AM-COM': 'C-ARGM-COM',
    'AM-LVB': 'ARGM-LVB',
    'R-AM-ADV': 'R-ARGM-ADV',
    'C-AM-ADV': 'C-ARGM-ADV',
    'R-AM-GOL': 'R-ARGM-GOL',
    'AM-MNR': 'ARGM-MNR',
    'C-AM-MNR': 'C-ARGM-MNR',
    'R-AM-DIR': 'R-ARGM-DIR',
    'AM-MOD': 'ARGM-MOD',
    'AM-CXN': 'ARGM-CXN',
    'C-AM-CXN': 'C-ARGM-CXN'
}

metadata_by_version_to_lang_to_treebank = {
    1: {
        'en': {
            'ewt': {
                'prefix_gold': '../resources/UniversalPropositions/UP-1.0/UP_English-EWT/',
                'prefix_pred': '../resources/UniversalPropositions/UP-1.0-predicted/UP_English-EWT/',
                'train': 'en_ewt-up-train.conllu',
                'dev': 'en_ewt-up-dev.conllu',
                'test': 'en_ewt-up-test.conllu',
                'load_ver': 2
            }
        },
        'fi': {
            'tdt': {
                'prefix_gold': '../resources/UniversalPropositions/UP-1.0/UP_Finnish/',
                'prefix_pred': '../resources/UniversalPropositions/UP-1.0-predicted/UP_Finnish/',
                'train': 'fi-up-train.conllu',
                'dev': 'fi-up-dev.conllu',
                'test': 'fi-up-test.conllu',
                'load_ver': 1
            }
        },
        'es': {
            'gsd': {
                'prefix_gold': '../resources/UniversalPropositions/UP-1.0/UP_Spanish/',
                'prefix_pred': '../resources/UniversalPropositions/UP-1.0-predicted/UP_Spanish/',
                'train': 'es-up-train.conllu',
                'dev': 'es-up-dev.conllu',
                'test': 'es-up-test.conllu',
                'load_ver': 1
            },
            'ancora': {
                'prefix_gold': '../resources/UniversalPropositions/UP-1.0/UP_Spanish-AnCora/',
                'prefix_pred': '../resources/UniversalPropositions/UP-1.0-predicted/UP_Spanish-AnCora/',
                'train': 'es_ancora-up-train.conllu',
                'dev': 'es_ancora-up-dev.conllu',
                'test': 'es_ancora-up-test.conllu',
                'load_ver': 1
            }
        },
        'fr': {
            'gsd': {
                'prefix_gold': '../resources/UniversalPropositions/UP-1.0/UP_French/',
                'prefix_pred': '../resources/UniversalPropositions/UP-1.0-predicted/UP_French/',
                'train': 'fr-up-train.conllu',
                'dev': 'fr-up-dev.conllu',
                'test': 'fr-up-test.conllu',
                'load_ver': 1
            }
        },
        'de': {
            'gsd': {
                'prefix_gold': '../resources/UniversalPropositions/UP-1.0/UP_German/',
                'prefix_pred': '../resources/UniversalPropositions/UP-1.0-predicted/UP_German/',
                'train': 'de-up-train.conllu',
                'dev': 'de-up-dev.conllu',
                'test': 'de-up-test.conllu',
                'load_ver': 1
            }
        },
        'it': {
            'isdt': {
                'prefix_gold': '../resources/UniversalPropositions/UP-1.0/UP_Italian/',
                'prefix_pred': '../resources/UniversalPropositions/UP-1.0-predicted/UP_Italian/',
                'train': 'it-up-train.conllu',
                'dev': 'it-up-dev.conllu',
                'test': 'it-up-test.conllu',
                'load_ver': 1
            }
        },
        'pt': {  # Train and dev are switched
            'bosque': {
                'prefix_gold': '../resources/UniversalPropositions/UP-1.0/UP_Portuguese-Bosque/',
                'prefix_pred': '../resources/UniversalPropositions/UP-1.0-predicted/UP_Portuguese-Bosque/',
                'train': 'pt_bosque-up-dev.conllu',
                'dev': 'pt_bosque-up-train.conllu',
                'test': 'pt_bosque-up-test.conllu',
                'load_ver': 1
            }
        },
        'zh': {
            'gsd': {
                'prefix_gold': '../resources/UniversalPropositions/UP-1.0/UP_Chinese/',
                'prefix_pred': '../resources/UniversalPropositions/UP-1.0-predicted/UP_Chinese/',
                'train': 'zh-up-train.conllu',
                'dev': 'zh-up-dev.conllu',
                'test': 'zh-up-test.conllu',
                'load_ver': 1
            }
        }
    },
    2: {
        'en': {
            'ewt': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_English-EWT/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_English-EWT/',
                'train': 'en_ewt-up-train.conllu',
                'dev': 'en_ewt-up-dev.conllu',
                'test': 'en_ewt-up-test.conllu',
                'load_ver': 2
            }
        },
        'fi': {
            'tdt': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Finnish-TDT/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Finnish-TDT/',
                'train': 'fi_tdt-ud-up-train.conllup',
                'dev': 'fi_tdt-ud-up-dev.conllup',
                'test': 'fi_tdt-ud-up-test.conllup',
                'load_ver': 3
            },
            'ftb': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Finnish-FTB/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Finnish-FTB/',
                'train': 'fi_ftb-ud-up-train.conllup',
                'dev': 'fi_ftb-ud-up-dev.conllup',
                'test': 'fi_ftb-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'fr': {
            'gsd': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_French-GSD/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_French-GSD/',
                'train': 'fr_gsd-ud-up-train.conllup',
                'dev': 'fr_gsd-ud-up-dev.conllup',
                'test': 'fr_gsd-ud-up-test.conllup',
                'load_ver': 3
            },
            'rhapsodie': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_French-Rhapsodie/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_French-Rhapsodie/',
                'train': 'fr_rhapsodie-ud-up-train.conllup',
                'dev': 'fr_rhapsodie-ud-up-dev.conllup',
                'test': 'fr_rhapsodie-ud-up-test.conllup',
                'load_ver': 3
            },
            'sequoia': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_French-Sequoia/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_French-Sequoia/',
                'train': 'fr_sequoia-ud-up-train.conllup',
                'dev': 'fr_sequoia-ud-up-dev.conllup',
                'test': 'fr_sequoia-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'de': {
            'gsd': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_German-GSD/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_German-GSD/',
                'train': 'de_gsd-ud-up-train.conllup',
                'dev': 'de_gsd-ud-up-dev.conllup',
                'test': 'de_gsd-ud-up-test.conllup',
                'load_ver': 3
            },
            'hdt': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_German-HDT/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_German-HDT/',
                'train': 'de_hdt-ud-up-train.conllup',
                'dev': 'de_hdt-ud-up-dev.conllup',
                'test': 'de_hdt-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'it': {
            'isdt': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Italian-ISDT/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Italian-ISDT/',
                'train': 'it_isdt-ud-up-train.conllup',
                'dev': 'it_isdt-ud-up-dev.conllup',
                'test': 'it_isdt-ud-up-test.conllup',
                'load_ver': 3
            },
            'partut': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Italian-ParTUT/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Italian-ParTUT/',
                'train': 'it_partut-ud-up-train.conllup',
                'dev': 'it_partut-ud-up-dev.conllup',
                'test': 'it_partut-ud-up-test.conllup',
                'load_ver': 3
            },
            'postwita': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Italian-PoSTWITA/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Italian-PoSTWITA/',
                'train': 'it_postwita-ud-up-train.conllup',
                'dev': 'it_postwita-ud-up-dev.conllup',
                'test': 'it_postwita-ud-up-test.conllup',
                'load_ver': 3
            },
            'twittiro': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Italian-TWITTIRO/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Italian-TWITTIRO/',
                'train': 'it_twittiro-ud-up-train.conllup',
                'dev': 'it_twittiro-ud-up-dev.conllup',
                'test': 'it_twittiro-ud-up-test.conllup',
                'load_ver': 3
            },
            'vit': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Italian-VIT/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Italian-VIT/',
                'train': 'it_vit-ud-up-train.conllup',
                'dev': 'it_vit-ud-up-dev.conllup',
                'test': 'it_vit-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'pt': {
            'bosque': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Portuguese-Bosque/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Portuguese-Bosque/',
                'train': 'pt_bosque-ud-up-train.conllup',
                'dev': 'pt_bosque-ud-up-dev.conllup',
                'test': 'pt_bosque-ud-up-test.conllup',
                'load_ver': 3
            },
            'gsd': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Portuguese-GSD/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Portuguese-GSD/',
                'train': 'pt_gsd-ud-up-train.conllup',
                'dev': 'pt_gsd-ud-up-dev.conllup',
                'test': 'pt_gsd-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'es': {
            'ancora': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Spanish-AnCora/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Spanish-AnCora/',
                'train': 'es_ancora-ud-up-train.conllup',
                'dev': 'es_ancora-ud-up-dev.conllup',
                'test': 'es_ancora-ud-up-test.conllup',
                'load_ver': 3
            },
            'gsd': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Spanish-GSD/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Spanish-GSD/',
                'train': 'es_gsd-ud-up-train.conllup',
                'dev': 'es_gsd-ud-up-dev.conllup',
                'test': 'es_gsd-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'cs': {
            'cltt': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Czech-CLTT/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Czech-CLTT/',
                'train': 'cs_cltt-ud-up-train.conllup',
                'dev': 'cs_cltt-ud-up-dev.conllup',
                'test': 'cs_cltt-ud-up-test.conllup',
                'load_ver': 3
            },
            'fictree': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Czech-FicTree/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Czech-FicTree/',
                'train': 'cs_fictree-ud-up-train.conllup',
                'dev': 'cs_fictree-ud-up-dev.conllup',
                'test': 'cs_fictree-ud-up-test.conllup',
                'load_ver': 3
            },
            'pdt': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Czech-PDT/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Czech-PDT/',
                'train': 'cs_pdt-ud-up-train.conllup',
                'dev': 'cs_pdt-ud-up-dev.conllup',
                'test': 'cs_pdt-ud-up-test.conllup',
                'load_ver': 3
            },
            'cac': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Czech-CAC/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Czech-CAC/',
                'train': 'cs_cac-ud-up-train.conllup',
                'dev': 'cs_cac-ud-up-dev.conllup',
                'test': 'cs_cac-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'el': {
            'gdt': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Greek-GDT/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Greek-GDT/',
                'train': 'el_gdt-ud-up-train.conllup',
                'dev': 'el_gdt-ud-up-dev.conllup',
                'test': 'el_gdt-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'ko': {
            'kaist': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Korean-Kaist/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Korean-Kaist/',
                'train': 'ko_kaist-ud-up-train.conllup',
                'dev': 'ko_kaist-ud-up-dev.conllup',
                'test': 'ko_kaist-ud-up-test.conllup',
                'load_ver': 3
            },
            'gsd': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Korean-GSD/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Korean-GSD/',
                'train': 'ko_gsd-ud-up-train.conllup',
                'dev': 'ko_gsd-ud-up-dev.conllup',
                'test': 'ko_gsd-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'ro': {
            'nonstandard': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Romanian-Nonstandard/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Romanian-Nonstandard/',
                'train': 'ro_nonstandard-ud-up-train.conllup',
                'dev': 'ro_nonstandard-ud-up-dev.conllup',
                'test': 'ro_nonstandard-ud-up-test.conllup',
                'load_ver': 3
            },
            'rrt': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Romanian-RRT/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Romanian-RRT/',
                'train': 'ro_rrt-ud-up-train.conllup',
                'dev': 'ro_rrt-ud-up-dev.conllup',
                'test': 'ro_rrt-ud-up-test.conllup',
                'load_ver': 3
            },
            'simonero': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Romanian-SiMoNERo/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Romanian-SiMoNERo/',
                'train': 'ro_simonero-ud-up-train.conllup',
                'dev': 'ro_simonero-ud-up-dev.conllup',
                'test': 'ro_simonero-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'hi': {
            'hdtb': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Hindi-HDTB/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Hindi-HDTB/',
                'train': 'hi_hdtb-ud-up-train.conllup',
                'dev': 'hi_hdtb-ud-up-dev.conllup',
                'test': 'hi_hdtb-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'mr': {
            'ufal': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Marathi-UFAL/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Marathi-UFAL/',
                'train': 'mr_ufal-ud-up-train.conllup',
                'dev': 'mr_ufal-ud-up-dev.conllup',
                'test': 'mr_ufal-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'ta': {
            'ttb': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Tamil-TTB/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Tamil-TTB/',
                'train': 'ta_ttb-ud-up-train.conllup',
                'dev': 'ta_ttb-ud-up-dev.conllup',
                'test': 'ta_ttb-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'hu': {
            'szeged': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Hungarian-Szeged/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Hungarian-Szeged/',
                'train': 'hu_szeged-ud-up-train.conllup',
                'dev': 'hu_szeged-ud-up-dev.conllup',
                'test': 'hu_szeged-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'pl': {
            'lfg': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Polish-LFG/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Polish-LFG/',
                'train': 'pl_lfg-ud-up-train.conllup',
                'dev': 'pl_lfg-ud-up-dev.conllup',
                'test': 'pl_lfg-ud-up-test.conllup',
                'load_ver': 3
            },
            'pdb': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Polish-PDB/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Polish-PDB/',
                'train': 'pl_pdb-ud-up-train.conllup',
                'dev': 'pl_pdb-ud-up-dev.conllup',
                'test': 'pl_pdb-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'te': {
            'mtg': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Telugu-MTG/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Telugu-MTG/',
                'train': 'te_mtg-ud-up-train.conllup',
                'dev': 'te_mtg-ud-up-dev.conllup',
                'test': 'te_mtg-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'nl': {
            'alpino': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Dutch-Alpino/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Dutch-Alpino/',
                'train': 'nl_alpino-ud-up-train.conllup',
                'dev': 'nl_alpino-ud-up-dev.conllup',
                'test': 'nl_alpino-ud-up-test.conllup',
                'load_ver': 3
            },
            'lassysmall': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Dutch-LassySmall/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Dutch-LassySmall/',
                'train': 'nl_lassysmall-ud-up-train.conllup',
                'dev': 'nl_lassysmall-ud-up-dev.conllup',
                'test': 'nl_lassysmall-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'id': {
            'gsd': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Indonesian-GSD/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Indonesian-GSD/',
                'train': 'id_gsd-ud-up-train.conllup',
                'dev': 'id_gsd-ud-up-dev.conllup',
                'test': 'id_gsd-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'ja': {
            'gsd': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Japanese-GSD/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Japanese-GSD/',
                'train': 'ja_gsd-ud-up-train.conllup',
                'dev': 'ja_gsd-ud-up-dev.conllup',
                'test': 'ja_gsd-ud-up-test.conllup',
                'load_ver': 3
            },
            'gsdluw': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Japanese-GSDLUW/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Japanese-GSDLUW/',
                'train': 'ja_gsdluw-ud-up-train.conllup',
                'dev': 'ja_gsdluw-ud-up-dev.conllup',
                'test': 'ja_gsdluw-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'ru': {
            'gsd': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Russian-GSD/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Russian-GSD/',
                'train': 'ru_gsd-ud-up-train.conllup',
                'dev': 'ru_gsd-ud-up-dev.conllup',
                'test': 'ru_gsd-ud-up-test.conllup',
                'load_ver': 3
            },
            'taiga': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Russian-Taiga/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Russian-Taiga/',
                'train': 'ru_taiga-ud-up-train.conllup',
                'dev': 'ru_taiga-ud-up-dev.conllup',
                'test': 'ru_taiga-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'uk': {
            'iu': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Ukrainian-IU/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Ukrainian-IU/',
                'train': 'uk_iu-ud-up-train.conllup',
                'dev': 'uk_iu-ud-up-dev.conllup',
                'test': 'uk_iu-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'zh': {
            'gsd': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Chinese-GSD/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Chinese-GSD/',
                'train': 'zh_gsd-ud-up-train.conllup',
                'dev': 'zh_gsd-ud-up-dev.conllup',
                'test': 'zh_gsd-ud-up-test.conllup',
                'load_ver': 3
            }
        },
        'vi': {
            'vtb': {
                'prefix_gold': '../resources/UniversalPropositions/UP-2.0/UP_Vietnamese-VTB/',
                'prefix_pred': '../resources/UniversalPropositions/UP-2.0-predicted/UP_Vietnamese-VTB/',
                'train': 'vi_vtb-ud-up-train.conllup',
                'dev': 'vi_vtb-ud-up-dev.conllup',
                'test': 'vi_vtb-ud-up-test.conllup',
                'load_ver': 3
            }
        }
    }
}