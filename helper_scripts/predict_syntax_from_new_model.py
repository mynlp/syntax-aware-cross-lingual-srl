import stanza
from stanza.utils.conll import CoNLL
import os
from utils.file import create_folder_if_not_exist

dir_map = {
    'UD_Japanese-GSDLUW': {
        'lang': 'ja',
        'package': 'gsd',
        'depparse_pretrain_path': '../stanza_resources/ja/pretrain/gsd.pt',
        'depparse_model_path': '../stanza/saved_models/depparse2/ja_gsdluw_parser.pt',
        'pos_pretrain_path': '../stanza_resources/ja/pretrain/gsd.pt',
        'pos_model_path': '../stanza/saved_models/pos2/ja_gsdluw_tagger.pt',
        'prefix': '../resources/ud-treebanks-v2.9'
    },
    'UD_French-Rhapsodie': {
        'lang': 'fr',
        'package': 'gsd',
        'depparse_pretrain_path': '../stanza_resources/fr/pretrain/gsd.pt',
        'depparse_model_path': '../stanza/saved_models/depparse2/fr_rhapsodie_parser.pt',
        'pos_pretrain_path': '../stanza_resources/fr/pretrain/gsd.pt',
        'pos_model_path': '../stanza/saved_models/pos2/fr_rhapsodie_tagger.pt',
        'prefix': '../resources/ud-treebanks-v2.9'
    },
    'UD_English-EWT': {
        'lang': 'en',
        'package': 'ewt',
        'depparse_pretrain_path': '../stanza_resources/en/pretrain/ewt.pt',
        'depparse_model_path': '../stanza/saved_models/depparse2/en_ewt_parser.pt',
        'pos_pretrain_path': '../stanza_resources/en/pretrain/ewt.pt',
        'pos_model_path': '../stanza/saved_models/pos2/en_ewt_tagger.pt',
        'prefix': '../resources/UniversalPropositions/UD-2.0-extracted'
    }
}


if __name__ == '__main__':
    for lang_code in dir_map:
        prefix = dir_map[lang_code]['prefix']
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
        depparse_pretrain_path = dir_map[lang_code]['depparse_pretrain_path']
        depparse_model_path = dir_map[lang_code]['depparse_model_path']
        pos_pretrain_path = dir_map[lang_code]['pos_pretrain_path']
        pos_model_path = dir_map[lang_code]['pos_model_path']

        nlp_depparse = stanza.Pipeline(
            lang=lang,
            processors='depparse',
            depparse_pretagged=True,
            dir='/work/gk77/k77022/stanza_resources',
            package=package,
            depparse_pretrain_path=depparse_pretrain_path,
            depparse_model_path=depparse_model_path
        )

        nlp_pos = stanza.Pipeline(
            lang=lang,
            processors='tokenize,pos',
            tokenize_pretokenized=True,
            dir='/work/gk77/k77022/stanza_resources',
            package=package,
            pos_pretrain_path=pos_pretrain_path,
            pos_model_path=pos_model_path
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
