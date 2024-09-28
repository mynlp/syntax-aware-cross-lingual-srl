import os
from utils.file import create_folder_if_not_exist


dict_map = {
    '../../resources/UniversalPropositions/UP-2.0/UP_English-EWT': {
        'source': '../../resources/UniversalPropositions/UD-2.0-extracted-predicted/UD_English-EWT',
        'target': '../../resources/UniversalPropositions/UP-2.0-predicted/UP_English-EWT'
    }
}


if __name__ == '__main__':
    for src_folder in dict_map:
        print(src_folder)
        src2_folder = dict_map[src_folder]['source']
        tgt_folder = dict_map[src_folder]['target']
        create_folder_if_not_exist(tgt_folder)
        files = os.listdir(src_folder)
        extracted_files = os.listdir(src2_folder)

        for file in files:
            if '.conllu' not in file or file.endswith('_.conllu') or 'propbank' in file:
                continue

            print(file)

            filename_src1 = f'{src_folder}/{file}'  # ori

            chosen_extracted_file = None

            if 'train' in file:
                chosen_extracted_file = [extracted_file for extracted_file in extracted_files if
                                         'train' in extracted_file]
            elif 'test' in file:
                chosen_extracted_file = [extracted_file for extracted_file in extracted_files if
                                         'test' in extracted_file]
            elif 'dev' in file:
                chosen_extracted_file = [extracted_file for extracted_file in extracted_files if
                                         'dev' in extracted_file]

            if chosen_extracted_file is None or len(chosen_extracted_file) != 1:
                raise Exception('Error.')

            filename_src2 = f'{src2_folder}/{chosen_extracted_file[0]}'  # extracted
            filename_tgt = f'{tgt_folder}/{file}'

            ori_comments = []
            ori_sentences = []

            file_src1 = open(filename_src1, 'r')

            line = file_src1.readline()

            while line:
                comment = []
                sentence = []

                while line and line.strip() != '':
                    sanitized_line = line.strip()

                    if sanitized_line[0] != '#':
                        sentence.append(sanitized_line.split('\t'))
                    else:
                        comment.append(f'{sanitized_line}\n')

                    line = file_src1.readline()

                while line and line.strip() == '':
                    line = file_src1.readline()

                if len(sentence) > 0:
                    ori_sentences.append(sentence)
                    ori_comments.append(comment)

            pred_sentences = []

            file_src2 = open(filename_src2, 'r')

            line = file_src2.readline()

            while line:
                sentence = []

                while line and line.strip() != '':
                    sanitized_line = line.strip()

                    if sanitized_line[0] != '#':
                        sentence.append(sanitized_line.split('\t'))

                    line = file_src2.readline()

                while line and line.strip() == '':
                    line = file_src2.readline()

                if len(sentence) > 0:
                    pred_sentences.append(sentence)

            if len(pred_sentences) != len(ori_sentences) or len(ori_sentences) != len(ori_comments):
                raise Exception(f'Different length. '
                                f'Pred: {len(pred_sentences)}. '
                                f'Ori: {len(ori_sentences)}. '
                                f'Comment: {len(ori_comments)}.')

            file_tgt = open(filename_tgt, 'w')

            for idx, ori_sentence in enumerate(ori_sentences):
                ori_valid_tokens = [word[0] for word in ori_sentence if '.' not in word[0]]
                pred_valid_tokens = [word[0] for word in pred_sentences[idx] if '.' not in word[0]]

                if len(ori_valid_tokens) != len(pred_valid_tokens):
                    print(ori_comments[idx])
                    raise Exception('Different length.')

                pred_inner_idx = 0

                for inner_idx, word in enumerate(ori_sentence):
                    if '.' not in word[0]:
                        word[3] = pred_sentences[idx][pred_inner_idx][3]  # upos
                        word[6] = pred_sentences[idx][pred_inner_idx][6]  # head
                        word[7] = pred_sentences[idx][pred_inner_idx][7]  # deprel

                        pred_inner_idx += 1

                    ori_sentence[inner_idx] = '\t'.join(ori_sentence[inner_idx])
                    ori_sentence[inner_idx] = f'{ori_sentence[inner_idx]}\n'

                file_tgt.writelines(ori_comments[idx])
                file_tgt.writelines(ori_sentence)
                file_tgt.write('\n')
