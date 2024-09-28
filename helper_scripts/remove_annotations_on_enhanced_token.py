import os
from utils import file, dataset


if __name__ == '__main__':
    folders = [
        'UP_Dutch-Alpino_fixed',
        'UP_Dutch-LassySmall_fixed',
        'UP_Ukrainian-IU_fixed',
        'UP_Finnish-TDT_fixed',
        'UP_Spanish-AnCora_fixed'
    ]

    for folder in folders:
        folder_path = f'../../resources/UniversalPropositions/{folder}'
        filenames = os.listdir(folder_path)
        filenames = [filename for filename in filenames if filename.endswith('.conllup')]
        assert len(filenames) == 3

        for filename in filenames:
            file_path = f'{folder_path}/{filename}'
            src = open(file_path, 'r')
            file.create_folder_if_not_exist(f'{folder_path}_final')
            dest = open(f'{folder_path}_final/{filename}', 'w')

            line = src.readline()

            count = 0

            while line:
                sent_block = []
                while line and line != '\n':
                    stripped_line = line.strip()
                    sent_block.append(stripped_line)
                    line = src.readline()

                comments = []
                all_token_attrs = []

                for sent_line in sent_block:
                    if dataset.is_comment(sent_line):
                        comments.append(f'{sent_line}\n')
                    else:
                        token_attrs = sent_line.split('\t')

                        assert len(token_attrs) == 4

                        if '.' in token_attrs[0]:  # enhanced
                            if token_attrs[1] != '_' or token_attrs[2] != '_' or token_attrs[3] != '_':
                                count += 1

                            for idx in range(1, len(token_attrs)):
                                token_attrs[idx] = '_'

                        all_token_attrs.append(token_attrs)

                dest.writelines(comments)
                for token_attrs in all_token_attrs:
                    lline = '\t'.join(token_attrs)
                    dest.write(f'{lline}\n')
                dest.write('\n')

                while line and line == '\n':
                    line = src.readline()

            print(filename, count)

            src.close()
            dest.close()
