import stanza
from stanza.utils.conll import CoNLL
import os
from utils.file import create_folder_if_not_exist

dir_map = {
    'UD_Czech-CLTT': {
        'lang': 'cs',
        'package': 'cltt'
    },
    'UD_Czech-FicTree': {
        'lang': 'cs',
        'package': 'fictree'
    },
    'UD_Czech-PDT': {
        'lang': 'cs',
        'package': 'pdt'
    },
    'UD_Czech-CAC': {
        'lang': 'cs',
        'package': 'cac'
    },
    'UD_Greek-GDT': {
        'lang': 'el',
        'package': 'gdt'
    },
    'UD_Korean-Kaist': {
        'lang': 'ko',
        'package': 'kaist'
    },
    'UD_Korean-GSD': {
        'lang': 'ko',
        'package': 'gsd'
    },
    'UD_Romanian-Nonstandard': {
        'lang': 'ro',
        'package': 'nonstandard'
    },
    'UD_Romanian-RRT': {
        'lang': 'ro',
        'package': 'rrt'
    },
    'UD_Romanian-SiMoNERo': {
        'lang': 'ro',
        'package': 'simonero'
    },
    'UD_Hindi-HDTB': {
        'lang': 'hi',
        'package': 'hdtb'
    },
    'UD_Marathi-UFAL': {
        'lang': 'mr',
        'package': 'ufal'
    },
    'UD_Tamil-TTB': {
        'lang': 'ta',
        'package': 'ttb'
    },
    'UD_Hungarian-Szeged': {
        'lang': 'hu',
        'package': 'szeged'
    },
    'UD_Polish-LFG': {
        'lang': 'pl',
        'package': 'lfg'
    },
    'UD_Polish-PDB': {
        'lang': 'pl',
        'package': 'pdb'
    },
    'UD_Telugu-MTG': {
        'lang': 'te',
        'package': 'mtg'
    },
    'UD_Dutch-Alpino': {
        'lang': 'nl',
        'package': 'alpino'
    },
    'UD_Dutch-LassySmall': {
        'lang': 'nl',
        'package': 'lassysmall'
    },
    'UD_Indonesian-GSD': {
        'lang': 'id',
        'package': 'gsd'
    },
    'UD_Japanese-GSD': {
        'lang': 'ja',
        'package': 'gsd'
    },
    # 'UD_Japanese-GSDLUW': {
    #     'lang': 'ja',
    #     'package': 'default'
    # },
    'UD_Russian-GSD': {
        'lang': 'ru',
        'package': 'gsd'
    },
    'UD_Russian-Taiga': {
        'lang': 'ru',
        'package': 'taiga'
    },
    'UD_Ukrainian-IU': {
        'lang': 'uk',
        'package': 'iu'
    },
    'UD_Chinese-GSD': {
        'lang': 'zh-hant',
        'package': 'gsd'
    },
    'UD_Vietnamese-VTB': {
        'lang': 'vi',
        'package': 'vtb'
    },
    # 'UD_English-EWT': {
    #     'lang': 'en',
    #     'package': 'ewt'
    # },
    'UD_Finnish-TDT': {
        'lang': 'fi',
        'package': 'tdt'
    },
    'UD_Finnish-FTB': {
        'lang': 'fi',
        'package': 'ftb'
    },
    'UD_Italian-ISDT': {
        'lang': 'it',
        'package': 'isdt'
    },
    'UD_Italian-ParTUT': {
        'lang': 'it',
        'package': 'partut'
    },
    'UD_Italian-PoSTWITA': {
        'lang': 'it',
        'package': 'postwita'
    },
    'UD_Italian-TWITTIRO': {
        'lang': 'it',
        'package': 'twittiro'
    },
    'UD_Italian-VIT': {
        'lang': 'it',
        'package': 'vit'
    },
    'UD_Spanish-GSD': {
        'lang': 'es',
        'package': 'gsd'
    },
    'UD_Spanish-AnCora': {
        'lang': 'es',
        'package': 'ancora'
    },
    'UD_French-GSD': {
        'lang': 'fr',
        'package': 'gsd'
    },
    # 'UD_French-Rhapsodie': {
    #     'lang': 'fr',
    #     'package': 'default'
    # },
    'UD_French-Sequoia': {
        'lang': 'fr',
        'package': 'sequoia'
    },
    'UD_German-GSD': {
        'lang': 'de',
        'package': 'gsd'
    },
    'UD_German-HDT': {
        'lang': 'de',
        'package': 'hdt'
    },
    'UD_Portuguese-Bosque': {
        'lang': 'pt',
        'package': 'bosque'
    },
    'UD_Portuguese-GSD': {
        'lang': 'pt',
        'package': 'gsd'
    }
}


if __name__ == '__main__':
    prefix = '../resources/ud-treebanks-v2.9'

    for lang_code in dir_map:
        src_folder_name = f'{prefix}/{lang_code}'
        dest_folder_name = f'{prefix}-predicted/{lang_code}'
        create_folder_if_not_exist(dest_folder_name)

        train_file = None
        dev_file = None
        test_file = None

        files = os.listdir(src_folder_name)

        for file in files:
            if 'conllu' in file:
                if 'train' in file:
                    train_file = file
                elif 'dev' in file:
                    dev_file = file
                elif 'test' in file:
                    test_file = file

        if not train_file or not dev_file or not test_file:
            raise Exception(f'File is not found. Train: {train_file}. Dev: {dev_file}. Test: {test_file}.')

        lang = dir_map[lang_code]['lang']
        package = dir_map[lang_code]['package']

        nlp_depparse = stanza.Pipeline(
            lang=lang,
            processors='depparse',
            depparse_pretagged=True,
            dir='/work/gk77/k77022/stanza_resources',
            package=package
        )

        nlp_pos = stanza.Pipeline(
            lang=lang,
            processors='tokenize,pos',
            tokenize_pretokenized=True,
            dir='/work/gk77/k77022/stanza_resources',
            package=package
        )

        for file in [train_file, dev_file, test_file]:
            input_file = f'{src_folder_name}/{file}'
            output_file = f'{dest_folder_name}/{file}'

            doc = CoNLL.conll2doc(input_file)

            dep_labeled_doc = nlp_depparse(doc)

            doc = CoNLL.conll2doc(input_file)

            pos_labeled_doc = nlp_pos(doc)

            assert len(pos_labeled_doc.sentences) == len(dep_labeled_doc.sentences)

            count = 0

            for idx_sent, sent in enumerate(dep_labeled_doc.sentences):
                if len(sent.words) != len(pos_labeled_doc.sentences[idx_sent].words):
                    count += 1
                    continue

                for idx_word, word in enumerate(sent.words):
                    ref = pos_labeled_doc.sentences[idx_sent].words[idx_word]
                    assert word.id == ref.id

                    word.upos = ref.upos

                    assert (word.lemma is not None) or (word.lemma == ref.lemma), print(file, pos_labeled_doc.sentences[idx_sent].words, sent.words)

            print(file, count)

            CoNLL.write_doc2conll(dep_labeled_doc, output_file)
