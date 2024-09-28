import os
from utils.file import create_folder_if_not_exist


dict_map = {
    '../../resources/UniversalPropositions/UP-2.0/UP_English-EWT': {
        'target': '../../resources/UniversalPropositions/UD-2.0-extracted/UD_English-EWT',
        'version': 2
    }
}


if __name__ == '__main__':
    for src_folder in dict_map:
        tgt_folder = dict_map[src_folder]['target']
        version = dict_map[src_folder]['version']
        create_folder_if_not_exist(tgt_folder)
        files = os.listdir(src_folder)

        for file in files:
            if '.conllu' not in file or file.endswith('_.conllu'):
                continue

            filename_src = f'{src_folder}/{file}'
            filename_tgt = f'{tgt_folder}/{file}'.replace('up', 'ud')

            sentences = []

            file_src = open(filename_src, 'r')

            line = file_src.readline()

            while line:
                sentence = []
                idx = 0
                start_idx = -1
                # raw_words = []

                while line and line.strip() != '':
                    sanitized_line = line.strip()

                    if sanitized_line[0] != '#':
                        if start_idx == -1:
                            start_idx = len(sentence)

                        words = sanitized_line.split('\t')

                        if '.' in words[0] and '-' in words[0]:
                            raise Exception(f'Token cannot be handled. ID: {words[0]}.')

                        if version == 1:
                            new_words = words[:8]

                            new_words.extend(['_', '_'])
                        else:
                            new_words = words[:10]

                        if new_words[5] != '_' and new_words[5].endswith('|'):
                            new_words[5] = new_words[5][:-1]

                        if len(new_words) != 10:
                            raise Exception(f'Length is not 10.')

                        new_line = '\t'.join(new_words)

                        sentence.append(f'{new_line}\n')
                    else:
                        sentence.append(f'{sanitized_line}\n')

                    line = file_src.readline()

                # if start_idx > -1:
                #     sentence.insert(start_idx, f'# text = {" ".join(raw_words)}\n')

                while line and line.strip() == '':
                    line = file_src.readline()

                if len(sentence) > 0:
                    sentences.append(sentence)

            file_tgt = open(filename_tgt, 'w')

            for sentence in sentences:
                file_tgt.writelines(sentence)
                file_tgt.write('\n')
